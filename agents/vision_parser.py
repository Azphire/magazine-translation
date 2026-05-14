import json
import logging
import os
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rapidocr_onnxruntime import RapidOCR
from PIL import Image

from config import PROMPT_DIR
from core.state import TranslationState
from utils.image_utils import encode_image_to_base64
from utils.layout_utils import guess_body_column, target_box_for_style, union_boxes
from utils.llm_client import get_gpt4o_client
from utils.logger import logger

logging.getLogger("rapidocr").setLevel(logging.ERROR)
ocr_engine = RapidOCR()

StyleName = Literal[
    "kicker",
    "title",
    "subtitle",
    "author",
    "body_heading",
    "body",
    "quote",
    "caption",
    "footer",
    "other",
]
AlignName = Literal["left", "center", "right", "justify"]


def load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


class TextBlock(BaseModel):
    id: int = Field(description="Stable block id in reading order")
    source_box: List[int] = Field(description="Logical source bbox [x1,y1,x2,y2] in absolute pixels")
    erase_boxes: List[List[int]] = Field(description="Every original OCR/PDF text line box that must be erased for this block")
    target_box: List[int] = Field(description="Recommended Chinese layout bbox [x1,y1,x2,y2]")
    text: str = Field(description="Exact merged English source text")
    style: StyleName = Field(description="Magazine semantic style")
    column: int = Field(default=0, description="Body column index, 0/1/2")
    reading_order: int = Field(description="Reading order, increasing")
    align: AlignName = Field(default="left")
    font_role: str = Field(default="body")
    color_role: str = Field(default="body")
    flow_id: str = Field(default="main", description="Article flow id. Same section/article should share flow_id.")


class ParsedLayout(BaseModel):
    blocks: List[TextBlock]


def extract_raw_ocr(img_path: str) -> list:
    result, _ = ocr_engine(img_path)
    ocr_blocks = []
    if not result:
        logger.warning("[Vision Parser] RapidOCR returned no results.")
        return ocr_blocks

    for idx, line in enumerate(result):
        box, text, score = line[0], line[1], line[2]
        x_min = int(min(pt[0] for pt in box))
        y_min = int(min(pt[1] for pt in box))
        x_max = int(max(pt[0] for pt in box))
        y_max = int(max(pt[1] for pt in box))
        ocr_blocks.append({
            "id": idx,
            "box": [x_min, y_min, x_max, y_max],
            "text": str(text),
            "score": round(float(score), 3),
            "source": "ocr",
        })
    return ocr_blocks


def _augment_blocks(parsed_json: dict, img_w: int, img_h: int, page_num: int, pdf_text_lines: list, raw_ocr: list) -> dict:
    """
    Makes VLM output robust: fills missing boxes/roles, keeps body columns, and uses PDF/OCR lines as erase boxes.
    """
    blocks = parsed_json.get("blocks", []) or []
    has_title = any(b.get("style") == "title" for b in blocks)
    all_lines = pdf_text_lines if pdf_text_lines else raw_ocr

    for idx, b in enumerate(blocks):
        b.setdefault("id", idx)
        b.setdefault("reading_order", idx)
        b.setdefault("style", "body")
        b.setdefault("text", "")
        style = b.get("style", "body")

        erase_boxes = b.get("erase_boxes") or []
        if not erase_boxes and b.get("source_box"):
            erase_boxes = [b["source_box"]]
        if not erase_boxes and b.get("box"):
            erase_boxes = [b["box"]]

        # If the VLM under-specifies erase boxes, supplement with text lines that are mostly inside source_box.
        source_box = b.get("source_box") or b.get("box") or union_boxes(erase_boxes)
        if source_box and all_lines:
            sx1, sy1, sx2, sy2 = source_box
            for line in all_lines:
                lb = line.get("box")
                if not lb:
                    continue
                cx = (lb[0] + lb[2]) / 2
                cy = (lb[1] + lb[3]) / 2
                if sx1 - 8 <= cx <= sx2 + 8 and sy1 - 8 <= cy <= sy2 + 8:
                    if lb not in erase_boxes:
                        erase_boxes.append(lb)

        source_box = b.get("source_box") or union_boxes(erase_boxes)
        b["source_box"] = [int(v) for v in source_box]
        b["erase_boxes"] = [[int(v) for v in eb] for eb in erase_boxes if eb and len(eb) == 4]

        if style == "body":
            b["column"] = int(b.get("column", guess_body_column(b["source_box"], img_w)))
            b["align"] = "justify"
            b.setdefault("font_role", "body")
            b.setdefault("color_role", "body")
        elif style == "body_heading":
            b["column"] = int(b.get("column", guess_body_column(b["source_box"], img_w)))
            b["align"] = "left"
            b.setdefault("font_role", "body_heading")
            b.setdefault("color_role", "subhead_red")
        elif style == "title":
            b.setdefault("font_role", "title")
            b.setdefault("color_role", "title_red")
            b["align"] = "left"
        elif style == "subtitle":
            b.setdefault("font_role", "subtitle")
            b.setdefault("color_role", "body")
            b["align"] = "left"
        elif style == "quote":
            b.setdefault("font_role", "quote")
            b.setdefault("color_role", "accent_green")
            b["align"] = "left"
        elif style == "author":
            b.setdefault("font_role", "author")
            b.setdefault("color_role", "muted")
            b["align"] = "right"
        else:
            b.setdefault("font_role", style)
            b.setdefault("color_role", "body")

        if not b.get("target_box") or b.get("target_box") == [0, 0, 0, 0]:
            b["target_box"] = target_box_for_style(style, img_w, img_h, page_num, has_title)

    blocks.sort(key=lambda x: int(x.get("reading_order", x.get("id", 0))))
    parsed_json["blocks"] = blocks
    return parsed_json


def vision_parser_node(state: TranslationState) -> dict:
    img_path = state.get("image_path")
    page_num = int(state.get("page_num", 0))
    retry_count = int(state.get("parser_retry_count", 0))
    errors = state.get("parser_errors")
    pdf_text_lines = state.get("pdf_text_lines", [])

    logger.info(f"[Vision Parser] OCR/PDF + VLM layout analysis: {img_path} page={page_num} retry={retry_count}")

    try:
        raw_ocr_data = extract_raw_ocr(img_path)
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        base64_image = encode_image_to_base64(img_path)
        llm = get_gpt4o_client(max_tokens=5000, temperature=0.1)
        structured_llm = llm.with_structured_output(ParsedLayout)
        sys_prompt = load_prompt("vision_parser.txt")

        if errors:
            sys_prompt += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:\n{errors}\nFix these layout defects."

        source_lines = pdf_text_lines if pdf_text_lines else raw_ocr_data
        message = HumanMessage(content=[
            {
                "type": "text",
                "text": json.dumps({
                    "page_num": page_num,
                    "image_size": [img_w, img_h],
                    "pdf_text_lines_if_available": pdf_text_lines,
                    "ocr_lines": raw_ocr_data,
                    "instruction": "Build a layout-aware ParsedLayout. Use PDF text boxes as the primary erase_boxes when available; use OCR as fallback. Never merge body text across visual columns. Mark red section titles as body_heading.",
                }, ensure_ascii=False),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ])

        response_data = structured_llm.invoke([SystemMessage(content=sys_prompt), message])
        parsed_json = response_data.model_dump()
        parsed_json = _augment_blocks(parsed_json, img_w, img_h, page_num, pdf_text_lines, raw_ocr_data)

        logger.info(f"[Vision Parser] Produced {len(parsed_json.get('blocks', []))} semantic blocks.")
        return {
            "raw_ocr": raw_ocr_data,
            "parsed_json": parsed_json,
            "parser_retry_count": retry_count + 1,
            "parser_errors": None,
        }

    except Exception as e:
        logger.error(f"[Vision Parser] Failed: {e}", exc_info=True)
        return {"parser_errors": str(e), "parser_retry_count": retry_count + 1}
