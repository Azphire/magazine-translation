from __future__ import annotations

import gc
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont

from agents.layout_engine import (
    DrawnBox,
    FlowFrame,
    build_body_atoms,
    choose_global_body_style,
    drawn_boxes_to_report,
    expected_body_refs,
    line_height,
    render_body_flow,
    text_width,
    wrap_text,
)
from config import (
    BODY_FIRST_LINE_INDENT_EM,
    BODY_FONT_SIZE,
    BODY_HEADING_GAP_RATIO,
    BODY_HEADING_SIZE_RATIO,
    BODY_LINE_GAP_CANDIDATES,
    BODY_MAX_SIZE_RATIO,
    BODY_MIN_SIZE_RATIO,
    BODY_PARAGRAPH_GAP_CANDIDATES,
    BODY_TARGET_FILL_RATIO,
    COLOR_ACCENT_GREEN,
    COLOR_BODY,
    COLOR_MUTED,
    COLOR_QUOTE_MARK,
    COLOR_SUBHEAD_RED,
    COLOR_TITLE_RED,
    DEBUG_SAVE_INTERMEDIATE,
    ERASE_BOX_PAD_RATIO,
    ERASE_INPAINT_RADIUS,
    ERASE_MAX_PAD,
    ERASE_MIN_PAD,
    CV_COMPONENT_MAX_HEIGHT_RATIO,
    CV_COMPONENT_MIN_AREA,
    CV_COMPONENT_MIN_WIDTH,
    CV_DARK_GRAY_THRESHOLD,
    CV_LINE_DILATE_H,
    CV_LINE_DILATE_W,
    CV_TEXT_DETECT_ENABLE,
    CV_WHITE_CONTEXT_THRESHOLD,
    FONT_BOLD,
    FONT_FALLBACK,
    FONT_MEDIUM,
    FONT_REGULAR,
    FONT_SERIF,
    OUTPUT_DIR,
    TEMP_DIR,
)
from core.state import TranslationState
from utils.layout_utils import clamp_box, infer_chinese_template
from utils.logger import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTHONMALLOC"] = "malloc"

logger.info("[Renderer] Initialized v4: document-level layout with deterministic layout critic support.")

_FONT_BYTES_CACHE: Dict[str, bytes] = {}
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font_path(role: str) -> str:
    candidates = {
        "title": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "subtitle": [FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "author": [FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "body": [FONT_SERIF, FONT_REGULAR, FONT_FALLBACK],
        "body_heading": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "heading": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "quote": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "caption": [FONT_REGULAR, FONT_FALLBACK],
        "footer": [FONT_REGULAR, FONT_FALLBACK],
        "regular": [FONT_REGULAR, FONT_FALLBACK],
        "medium": [FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "bold": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
    }
    for path in candidates.get(role, [FONT_REGULAR, FONT_FALLBACK]):
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No usable CJK font found. Put fonts into data/fonts, including simhei.ttf as fallback."
    )


def get_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    size = max(6, int(size))
    path = _font_path(role)
    key = (path, size)
    if key not in _FONT_CACHE:
        if path not in _FONT_BYTES_CACHE:
            with open(path, "rb") as f:
                _FONT_BYTES_CACHE[path] = f.read()
        _FONT_CACHE[key] = ImageFont.truetype(io.BytesIO(_FONT_BYTES_CACHE[path]), size)
    return _FONT_CACHE[key]


def _box_pad(box: List[int]) -> int:
    height = max(1, box[3] - box[1])
    return max(ERASE_MIN_PAD, min(ERASE_MAX_PAD, int(height * ERASE_BOX_PAD_RATIO)))


def _iou(a: List[int], b: List[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, area_a + area_b - inter)


def detect_cv_text_boxes(raw_img: Image.Image, existing_boxes: List[List[int]], basename: str) -> List[List[int]]:
    """
    Detect non-selectable or OCR-missed text on white paper.

    The detector is used only for erasing. It is intentionally conservative around
    photo regions and broad graphic elements.
    """
    if not CV_TEXT_DETECT_ENABLE:
        return []

    arr = np.array(raw_img.convert("RGB"))
    img_h, img_w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    white_pixels = (gray > 238).astype(np.uint8)
    k = max(21, int(min(img_w, img_h) * 0.018))
    if k % 2 == 0:
        k += 1
    white_context = cv2.blur(white_pixels.astype(np.float32), (k, k)) > float(CV_WHITE_CONTEXT_THRESHOLD)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturated_colored = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 245)
    darkish = gray < int(CV_DARK_GRAY_THRESHOLD)
    candidate = ((darkish | saturated_colored) & white_context).astype(np.uint8) * 255
    candidate = cv2.medianBlur(candidate, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(CV_LINE_DILATE_W), int(CV_LINE_DILATE_H)))
    line_mask = cv2.dilate(candidate, kernel, iterations=1)
    num, _, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)

    boxes: List[List[int]] = []
    max_h = max(8, int(img_h * CV_COMPONENT_MAX_HEIGHT_RATIO))
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < CV_COMPONENT_MIN_AREA:
            continue
        if w < CV_COMPONENT_MIN_WIDTH or h < 3 or h > max_h:
            continue
        if w > img_w * 0.78 or h > img_h * 0.10:
            continue
        box = [int(x), int(y), int(x + w), int(y + h)]
        crop = candidate[y:y + h, x:x + w]
        if crop.size == 0 or int(np.count_nonzero(crop)) < 3:
            continue
        if any(_iou(box, eb) > 0.72 for eb in existing_boxes):
            continue
        boxes.append(box)

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = []
    for box in boxes:
        if merged and not (box[0] > merged[-1][2] + 20 or box[1] > merged[-1][3] + 8):
            prev = merged[-1]
            merged[-1] = [min(prev[0], box[0]), min(prev[1], box[1]), max(prev[2], box[2]), max(prev[3], box[3])]
        else:
            merged.append(box)

    if DEBUG_SAVE_INTERMEDIATE:
        os.makedirs(TEMP_DIR, exist_ok=True)
        Image.fromarray(candidate).save(os.path.join(TEMP_DIR, f"{basename}_cv_text_candidates.png"))
        Image.fromarray(line_mask).save(os.path.join(TEMP_DIR, f"{basename}_cv_text_lines.png"))

    logger.info(f"[Renderer] CV fallback detected {len(merged)} extra text-line boxes for erasing.")
    return merged


def collect_erase_boxes(state: TranslationState, img_w: int, img_h: int) -> List[List[int]]:
    boxes: List[List[int]] = []

    for source_name in ("pdf_text_lines", "raw_ocr", "cv_text_lines"):
        for line in state.get(source_name, []) or []:
            box = line.get("box")
            clamped = clamp_box(box, img_w, img_h) if box else None
            if clamped:
                boxes.append(clamped)

    for block in state.get("translated_blocks", []) or []:
        for box in block.get("erase_boxes") or []:
            clamped = clamp_box(box, img_w, img_h)
            if clamped:
                boxes.append(clamped)
        for key in ("source_box", "box"):
            box = block.get(key)
            clamped = clamp_box(box, img_w, img_h) if box else None
            if clamped:
                boxes.append(clamped)

    parsed_blocks = (state.get("parsed_json") or {}).get("blocks", []) or []
    for block in parsed_blocks:
        for box in block.get("erase_boxes") or []:
            clamped = clamp_box(box, img_w, img_h)
            if clamped:
                boxes.append(clamped)
        box = block.get("source_box") or block.get("box")
        clamped = clamp_box(box, img_w, img_h) if box else None
        if clamped:
            boxes.append(clamped)

    seen = set()
    unique: List[List[int]] = []
    for box in boxes:
        key = tuple(int(v // 3) for v in box)
        if key not in seen:
            seen.add(key)
            unique.append(box)
    return unique


def erase_original_text(raw_img: Image.Image, erase_boxes: List[List[int]], basename: str) -> Tuple[Image.Image, Dict[str, str]]:
    arr = np.array(raw_img.convert("RGB"))
    img_h, img_w = arr.shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for box in erase_boxes:
        pad = _box_pad(box)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(arr, mask, ERASE_INPAINT_RADIUS, cv2.INPAINT_TELEA)

    for box in erase_boxes:
        pad = _box_pad(box)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
        roi = arr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        flat = roi.reshape(-1, 3)
        median = np.median(flat, axis=0)
        if float(np.mean(median)) > 222 and float(np.std(flat)) < 58:
            inpainted[y1:y2, x1:x2] = np.array([255, 255, 255], dtype=np.uint8)

    debug_paths: Dict[str, str] = {}
    if DEBUG_SAVE_INTERMEDIATE:
        os.makedirs(TEMP_DIR, exist_ok=True)
        mask_path = os.path.join(TEMP_DIR, f"{basename}_erase_mask.png")
        inpaint_path = os.path.join(TEMP_DIR, f"{basename}_inpainted_before_render.png")
        Image.fromarray(mask).save(mask_path)
        Image.fromarray(inpainted).save(inpaint_path)
        debug_paths["erase_mask"] = mask_path
        debug_paths["inpainted_before_render"] = inpaint_path

    return Image.fromarray(inpainted), debug_paths


def _sorted_blocks(state: TranslationState) -> List[dict]:
    return sorted(
        state.get("translated_blocks", []) or [],
        key=lambda b: (int(b.get("reading_order", b.get("id", 0))), int(b.get("id", 0))),
    )


def _make_drawn_box(page_index: int, rect: List[int], role: str, font_size: int, block: dict, text: str) -> DrawnBox:
    page_num = int(block.get("page_num", page_index))
    ref = str(block.get("global_key") or f"p{page_num}_b{block.get('id')}")
    return DrawnBox(page_index=page_index, rect=rect, role=role, font_size=font_size, source_ids=[ref], text=text)


def draw_fixed_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: List[int],
    role: str,
    max_size: int,
    min_size: int,
    fill: Tuple[int, int, int],
    align: str = "left",
    line_gap_ratio: float = 0.25,
) -> Tuple[List[List[int]], int]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [], min_size

    width = max(10, box[2] - box[0])
    height = max(10, box[3] - box[1])
    chosen_font: Optional[ImageFont.FreeTypeFont] = None
    chosen_lines: List[str] = []
    chosen_size = min_size
    chosen_gap = max(1, int(min_size * line_gap_ratio))

    for size in range(max_size, min_size - 1, -1):
        font = get_font(role, size)
        gap = max(1, int(size * line_gap_ratio))
        lines = wrap_text(text, font, width)
        needed = len(lines) * line_height(font) + max(0, len(lines) - 1) * gap
        if needed <= height:
            chosen_font = font
            chosen_lines = lines
            chosen_size = size
            chosen_gap = gap
            break

    if chosen_font is None:
        chosen_font = get_font(role, min_size)
        chosen_lines = wrap_text(text, chosen_font, width)

    rects: List[List[int]] = []
    y = box[1]
    lh = line_height(chosen_font)
    for line in chosen_lines:
        if y + lh > box[3]:
            break
        lw = text_width(chosen_font, line)
        if align == "center":
            x = box[0] + max(0, (width - lw) // 2)
        elif align == "right":
            x = box[2] - lw
        else:
            x = box[0]
        draw.text((x, y), line, font=chosen_font, fill=fill)
        rects.append([x, y, x + lw, y + lh])
        y += lh + chosen_gap
    return rects, chosen_size


def render_non_body_elements(
    draw: ImageDraw.ImageDraw,
    blocks: List[dict],
    img_w: int,
    img_h: int,
    page_index: int,
) -> List[DrawnBox]:
    drawn: List[DrawnBox] = []
    has_title = any(b.get("style") == "title" for b in blocks)
    template = infer_chinese_template(img_w, img_h, page_index, has_title)

    by_style = {
        style: [b for b in blocks if b.get("style") == style]
        for style in ["kicker", "title", "subtitle", "author", "quote", "caption", "footer"]
    }

    for block in by_style.get("kicker", []):
        box = clamp_box(block.get("target_box") or template.get("kicker"), img_w, img_h)
        if not box:
            continue
        font_size = max(10, int(img_w * 0.010))
        font = get_font("caption", font_size)
        text = block.get("target_text", "")
        draw.text((box[0], box[1]), text, font=font, fill=COLOR_MUTED)
        x = box[0] + text_width(font, text) + 5
        y = box[1] + 3
        draw.polygon([(x, y), (x + 12, y + 8), (x, y + 16)], fill=COLOR_ACCENT_GREEN)
        drawn.append(_make_drawn_box(page_index, [box[0], box[1], box[2], box[1] + line_height(font)], "kicker", font_size, block, text))

    for block in by_style.get("title", []):
        box = clamp_box(block.get("target_box") or template.get("title"), img_w, img_h)
        if not box:
            continue
        rects, font_size = draw_fixed_text(
            draw,
            block.get("target_text", ""),
            box,
            "title",
            int(img_h * 0.058),
            int(img_h * 0.040),
            COLOR_TITLE_RED,
            align="left",
            line_gap_ratio=0.08,
        )
        drawn.extend(_make_drawn_box(page_index, rect, "title", font_size, block, block.get("target_text", "")) for rect in rects)

    for block in by_style.get("subtitle", []):
        box = clamp_box(block.get("target_box") or template.get("subtitle"), img_w, img_h)
        if not box:
            continue
        rects, font_size = draw_fixed_text(
            draw,
            block.get("target_text", ""),
            box,
            "subtitle",
            int(img_h * 0.022),
            int(img_h * 0.017),
            COLOR_BODY,
            align="left",
            line_gap_ratio=0.42,
        )
        drawn.extend(_make_drawn_box(page_index, rect, "subtitle", font_size, block, block.get("target_text", "")) for rect in rects)

    if by_style.get("author"):
        box = clamp_box(template.get("authors") or by_style["author"][0].get("target_box"), img_w, img_h)
        if box:
            y = box[1]
            name_font_size = int(img_h * 0.012)
            bio_font_size = int(img_h * 0.011)
            name_font = get_font("author", name_font_size)
            bio_font = get_font("caption", bio_font_size)
            for block in by_style["author"]:
                text = block.get("target_text", "")
                parts = [p.strip() for p in re.split(r"[\n；;]+", text) if p.strip()]
                if not parts:
                    continue
                name = parts[0]
                bio = "；".join(parts[1:]) if len(parts) > 1 else ""
                name_w = text_width(name_font, name)
                draw.text((box[2] - name_w, y), name, font=name_font, fill=COLOR_BODY)
                drawn.append(_make_drawn_box(page_index, [box[2] - name_w, y, box[2], y + line_height(name_font)], "author", name_font_size, block, name))
                y += line_height(name_font) + 5
                draw.line((box[0], y, box[2], y), fill=COLOR_ACCENT_GREEN, width=1)
                y += 7
                for line in wrap_text(bio, bio_font, box[2] - box[0]):
                    line_w = text_width(bio_font, line)
                    draw.text((box[2] - line_w, y), line, font=bio_font, fill=COLOR_ACCENT_GREEN)
                    drawn.append(_make_drawn_box(page_index, [box[2] - line_w, y, box[2], y + line_height(bio_font)], "author", bio_font_size, block, line))
                    y += line_height(bio_font) + 3
                y += int(img_h * 0.030)

    for block in by_style.get("caption", []):
        box = clamp_box(block.get("target_box") or template.get("caption"), img_w, img_h)
        if not box:
            continue
        rects, font_size = draw_fixed_text(
            draw,
            block.get("target_text", ""),
            box,
            "caption",
            int(img_h * 0.0125),
            int(img_h * 0.0095),
            COLOR_BODY,
            align="left",
            line_gap_ratio=0.18,
        )
        drawn.extend(_make_drawn_box(page_index, rect, "caption", font_size, block, block.get("target_text", "")) for rect in rects)

    for block in by_style.get("quote", []):
        box = clamp_box(block.get("target_box") or template.get("quote"), img_w, img_h)
        if not box:
            continue
        mark_size = int(img_h * 0.050)
        draw.text((box[0], max(0, box[1] - int(img_h * 0.025))), "“", font=get_font("quote", mark_size), fill=COLOR_QUOTE_MARK)
        rects, font_size = draw_fixed_text(
            draw,
            block.get("target_text", ""),
            box,
            "quote",
            int(img_h * 0.034),
            int(img_h * 0.024),
            COLOR_ACCENT_GREEN,
            align="left",
            line_gap_ratio=0.28,
        )
        drawn.extend(_make_drawn_box(page_index, rect, "quote", font_size, block, block.get("target_text", "")) for rect in rects)

    return drawn


def get_body_frames_for_page(page_index: int, img_w: int, img_h: int, has_title: bool) -> List[FlowFrame]:
    template = infer_chinese_template(img_w, img_h, page_index, has_title)
    frames: List[FlowFrame] = []
    for box in template.get("body_columns", []):  # type: ignore[union-attr]
        clamped = clamp_box(box, img_w, img_h)
        if clamped:
            frames.append(FlowFrame(page_index=page_index, box=clamped, role="body"))
    return frames


def prepare_clean_page(state: TranslationState, page_index: int) -> Tuple[Image.Image, Dict[str, str]]:
    img_path = state.get("image_path")
    if not img_path or not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing page image: {img_path}")
    with Image.open(img_path) as im:
        raw_img = im.convert("RGB")
    img_w, img_h = raw_img.size
    basename = os.path.splitext(os.path.basename(img_path))[0]
    erase_boxes = collect_erase_boxes(state, img_w, img_h)
    cv_extra = detect_cv_text_boxes(raw_img, erase_boxes, basename)
    if cv_extra:
        state["cv_text_lines"] = [{"id": i, "box": b, "text": "", "source": "cv"} for i, b in enumerate(cv_extra)]
        erase_boxes = collect_erase_boxes(state, img_w, img_h)
    logger.info(f"[Renderer] Page {page_index}: erasing {len(erase_boxes)} text boxes.")
    return erase_original_text(raw_img, erase_boxes, basename)


def render_document_pages(
    page_states: List[TranslationState],
    document_translation: Dict[str, Any],
    layout_suggestions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render the whole document at once so body typography is globally consistent."""
    layout_suggestions = layout_suggestions or {}
    images: List[Image.Image] = []
    draw_by_page: Dict[int, ImageDraw.ImageDraw] = {}
    debug_paths: Dict[int, Dict[str, str]] = {}
    body_frames: List[FlowFrame] = []
    non_body_drawn: List[DrawnBox] = []

    for page_index, state in enumerate(page_states):
        clean_img, page_debug = prepare_clean_page(state, page_index)
        images.append(clean_img)
        draw_by_page[page_index] = ImageDraw.Draw(clean_img)
        debug_paths[page_index] = page_debug

    if not images:
        return {"output_image_paths": [], "layout_report": {}, "debug_paths": debug_paths}

    for page_index, state in enumerate(page_states):
        img_w, img_h = images[page_index].size
        blocks = _sorted_blocks(state)
        has_title = any(b.get("style") == "title" for b in blocks)
        non_body_drawn.extend(render_non_body_elements(draw_by_page[page_index], blocks, img_w, img_h, page_index))
        body_frames.extend(get_body_frames_for_page(page_index, img_w, img_h, has_title))

    img_w, img_h = images[0].size
    atoms = build_body_atoms(page_states, document_translation)
    refs = expected_body_refs(page_states)

    config_values = {
        "BODY_FONT_SIZE": BODY_FONT_SIZE,
        "BODY_MIN_SIZE_RATIO": BODY_MIN_SIZE_RATIO,
        "BODY_MAX_SIZE_RATIO": BODY_MAX_SIZE_RATIO,
        "BODY_TARGET_FILL_RATIO": BODY_TARGET_FILL_RATIO,
        "BODY_LINE_GAP_CANDIDATES": BODY_LINE_GAP_CANDIDATES,
        "BODY_PARAGRAPH_GAP_CANDIDATES": BODY_PARAGRAPH_GAP_CANDIDATES,
        "BODY_HEADING_SIZE_RATIO": BODY_HEADING_SIZE_RATIO,
        "BODY_HEADING_GAP_RATIO": BODY_HEADING_GAP_RATIO,
        "BODY_FIRST_LINE_INDENT_EM": BODY_FIRST_LINE_INDENT_EM,
    }
    style = choose_global_body_style(atoms, body_frames, get_font, img_h, config_values, layout_suggestions)
    logger.info(
        f"[Renderer] Global body style: size={style.body_font_size}, "
        f"line_gap={style.body_line_gap_ratio:.2f}, para_gap={style.body_para_gap_ratio:.2f}"
    )

    body_drawn, unrendered_chars, page_usage = render_body_flow(
        draw_by_page=draw_by_page,
        atoms=atoms,
        frames=body_frames,
        font_loader=get_font,
        style=style,
        fill=COLOR_BODY,
        heading_fill=COLOR_SUBHEAD_RED,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_paths: List[str] = []
    for page_index, image in enumerate(images):
        out_path = os.path.join(OUTPUT_DIR, f"final_page_{page_index}.jpg")
        image.save(out_path, quality=95)
        output_paths.append(out_path)

    layout_report = drawn_boxes_to_report(
        drawn=body_drawn + non_body_drawn,
        frames=body_frames,
        page_usage=page_usage,
        unrendered_body_chars=unrendered_chars,
        expected_refs=refs,
    )

    for image in images:
        image.close()
    gc.collect()

    return {
        "output_image_paths": output_paths,
        "layout_report": layout_report,
        "debug_paths": debug_paths,
        "global_body_style": {
            "font_size": style.body_font_size,
            "line_gap_ratio": style.body_line_gap_ratio,
            "para_gap_ratio": style.body_para_gap_ratio,
        },
    }


def renderer_node(state: TranslationState) -> dict:
    """Compatibility wrapper for single-page rendering."""
    result = render_document_pages([state], document_translation=state.get("document_translation", {}) or {})
    paths = result.get("output_image_paths", [])
    return {
        "output_image_path": paths[0] if paths else None,
        "debug_paths": result.get("debug_paths", {}).get(0, {}),
        "layout_report": result.get("layout_report", {}),
    }
