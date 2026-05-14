from __future__ import annotations

import os
from typing import Dict, List, Optional

import fitz
from PIL import Image

from agents.critics import parser_critic_node
from agents.document_translator import translate_document
from agents.memory import MemoryAgent
from agents.renderer import renderer_node
from agents.vision_parser import vision_parser_node
from config import MAX_RETRIES, OUTPUT_DIR, PDF_RENDER_ZOOM, TEMP_DIR
from core.state import TranslationState
from utils.logger import logger


def extract_pdf_text_lines(page: fitz.Page, zoom: float) -> List[Dict]:
    """
    Extract vector text boxes directly from the PDF.

    v3 improvement:
    - keep both line boxes and span boxes;
    - scale coordinates to rendered image pixels;
    - these boxes are used only for erasing/layout coverage, not translation.
    """
    lines: List[Dict] = []
    data = page.get_text("dict")
    idx = 0
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
                    "id": idx,
                    "box": [int(round(x1 * zoom)), int(round(y1 * zoom)), int(round(x2 * zoom)), int(round(y2 * zoom))],
                    "text": text,
                    "source": "pdf_line",
                })
                idx += 1
            for span in spans:
                stext = str(span.get("text", "")).strip()
                sb = span.get("bbox")
                if not stext or not sb:
                    continue
                x1, y1, x2, y2 = sb
                # Span boxes are useful when line boxes are too loose or line extraction misses small fragments.
                lines.append({
                    "id": idx,
                    "box": [int(round(x1 * zoom)), int(round(y1 * zoom)), int(round(x2 * zoom)), int(round(y2 * zoom))],
                    "text": stext,
                    "source": "pdf_span",
                })
                idx += 1
    logger.info(f"[PDF Text] Extracted {len(lines)} vector line/span boxes.")
    return lines


def render_pdf_page_to_image(page: fitz.Page, page_num: int, temp_dir: str, zoom: float) -> str:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
    pix.save(img_path)
    return img_path


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
    """
    Robust PDF stitching. Avoid PIL multi-page edge cases by inserting every image
    into a new PyMuPDF page explicitly.
    """
    if not image_paths:
        raise ValueError("No images to stitch into PDF.")
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    pdf = fitz.open()
    for img_path in image_paths:
        if not img_path or not os.path.exists(img_path):
            logger.warning(f"[PDF Stitch] Missing output image skipped: {img_path}")
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        page = pdf.new_page(width=w, height=h)
        page.insert_image(fitz.Rect(0, 0, w, h), filename=img_path)
    if pdf.page_count == 0:
        raise ValueError("PDF stitch failed: no valid image pages were inserted.")
    pdf.save(output_pdf_path, deflate=True)
    pdf.close()
    logger.info(f"✅ Stitched {len(image_paths)} image pages into PDF: {output_pdf_path}")


def parse_pdf_pages(doc: fitz.Document, memory_agent: MemoryAgent) -> List[TranslationState]:
    page_states: List[TranslationState] = []
    for page_num in range(len(doc)):
        logger.info(f"\n--- Parse page {page_num + 1}/{len(doc)} ---")
        page = doc.load_page(page_num)
        img_path = render_pdf_page_to_image(page, page_num, TEMP_DIR, PDF_RENDER_ZOOM)
        pdf_text_lines = extract_pdf_text_lines(page, PDF_RENDER_ZOOM)

        state = TranslationState(
            image_path=img_path,
            page_num=page_num,
            pdf_text_lines=pdf_text_lines,
            memory_dict=memory_agent.get_memory_context(),
            parser_retry_count=0,
            translator_retry_count=0,
            flow_context={},
        )
        state = run_parser_with_retries(state)
        if state.get("parser_errors"):
            logger.error(f"[Page {page_num}] Parser failed but page will still be rendered if possible: {state.get('parser_errors')}")
        page_states.append(state)
    return page_states


def render_translated_pages(page_states: List[TranslationState], document_translation: Dict) -> List[str]:
    output_images: List[str] = []
    flow_context: Dict = {
        "use_global_body_flow": bool(document_translation.get("use_global_body_flow")),
        "global_body_queue": list(document_translation.get("global_body_flow") or []),
    }

    for state in page_states:
        page_num = int(state.get("page_num", 0))
        logger.info(f"\n--- Render page {page_num + 1}/{len(page_states)} ---")
        state["flow_context"] = flow_context
        result = renderer_node(state)
        state.update(result)
        flow_context = state.get("flow_context", flow_context)
        out = state.get("output_image_path")
        if out and os.path.exists(out):
            output_images.append(out)
        else:
            logger.error(f"[Render] Page {page_num} did not produce an output image.")

    if flow_context.get("body_overflow_elements"):
        logger.warning(
            f"[Flow] Unrendered body overflow after last page: "
            f"{len(flow_context['body_overflow_elements'])} elements."
        )
    return output_images


def update_memory_from_document(memory_agent: MemoryAgent, page_states: List[TranslationState], document_translation: Dict) -> None:
    pairs = []
    for state in page_states:
        for b in state.get("translated_blocks", []) or []:
            if b.get("source_text") and b.get("target_text"):
                pairs.append(b)
    # Add synthetic body pairs from global flow so terminology memory sees cross-page body translation.
    for i, elem in enumerate(document_translation.get("global_body_flow") or []):
        pairs.append({"source_text": " / ".join(elem.get("source_refs", [])), "target_text": elem.get("text", "")})
    if pairs:
        memory_agent.update_memory(pairs)


def process_pdf(pdf_path: str, output_pdf_path: str):
    logger.info(f"=== Starting Magazine Translation Pipeline v3: {pdf_path} ===")
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    memory_agent = MemoryAgent()
    doc = fitz.open(pdf_path)
    page_states = parse_pdf_pages(doc, memory_agent)

    logger.info("\n=== Document-level translation: judging cross-page continuity and translating global body flow ===")
    page_states, document_translation = translate_document(page_states, memory_agent.get_memory_context(), max_retries=MAX_RETRIES)

    output_images = render_translated_pages(page_states, document_translation)
    update_memory_from_document(memory_agent, page_states, document_translation)

    if output_images:
        build_pdf_from_images(output_images, output_pdf_path)
    else:
        logger.error("No output pages were generated.")


def process_sample_images(image_paths: List[str], output_dir: Optional[str] = None):
    """
    Debug helper for already-rendered page images. It cannot use PDF-native boxes,
    so English erasing falls back to OCR + CV text-line detection.
    """
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    memory_agent = MemoryAgent()
    page_states: List[TranslationState] = []
    for page_num, img_path in enumerate(image_paths):
        state = TranslationState(
            image_path=img_path,
            page_num=page_num,
            pdf_text_lines=[],
            memory_dict=memory_agent.get_memory_context(),
            parser_retry_count=0,
            translator_retry_count=0,
            flow_context={},
        )
        page_states.append(run_parser_with_retries(state))

    page_states, document_translation = translate_document(page_states, memory_agent.get_memory_context(), max_retries=MAX_RETRIES)
    render_translated_pages(page_states, document_translation)


if __name__ == "__main__":
    input_pdf = "./data/input/magazine.pdf"
    output_pdf = "./data/output/magazine_zh.pdf"
    process_pdf(input_pdf, output_pdf)
