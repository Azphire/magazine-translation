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

logger.info("[Renderer] Initialized v6: completeness-aware document layout with flexible non-body placement.")

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
) -> Tuple[List[List[int]], int, bool]:
    """Draw wrapped text and report whether any lines were clipped.

    Earlier versions only returned the visible rectangles. That made pull quotes,
    deck text, and captions look "rendered" even when most lines had been
    clipped. The third return value lets callers and critics treat clipping as a
    layout failure instead of a successful render.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [], min_size, False

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
        chosen_size = min_size
        chosen_gap = max(1, int(min_size * line_gap_ratio))

    rects: List[List[int]] = []
    y = box[1]
    lh = line_height(chosen_font)
    drawn_lines = 0
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
        drawn_lines += 1
        y += lh + chosen_gap
    return rects, chosen_size, drawn_lines < len(chosen_lines)


def _measure_text_candidate(
    text: str,
    box: List[int],
    role: str,
    max_size: int,
    min_size: int,
    line_gap_ratio: float,
) -> Dict[str, Any]:
    """Measure a candidate box without drawing."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    width = max(10, box[2] - box[0])
    height = max(10, box[3] - box[1])
    best = {"font_size": min_size, "overflow_px": 10**9, "hidden_lines": 10**6, "line_count": 0}
    for size in range(max_size, min_size - 1, -1):
        font = get_font(role, size)
        gap = max(1, int(size * line_gap_ratio))
        lines = wrap_text(text, font, width)
        lh = line_height(font)
        needed = len(lines) * lh + max(0, len(lines) - 1) * gap
        visible = max(0, (height + gap) // max(1, lh + gap))
        overflow = max(0, needed - height)
        hidden = max(0, len(lines) - visible)
        current = {"font_size": size, "overflow_px": overflow, "hidden_lines": hidden, "line_count": len(lines)}
        if overflow == 0:
            return current
        if overflow < best["overflow_px"] or (overflow == best["overflow_px"] and size > best["font_size"]):
            best = current
    return best


def choose_flexible_text_box(
    text: str,
    preferred: List[int],
    candidates: List[List[int]],
    occupied: List[List[int]],
    role: str,
    max_size: int,
    min_size: int,
    line_gap_ratio: float,
) -> List[int]:
    """Choose a box by combining fit quality and collision score.

    This generic routine is used for deck text, pull quotes, and captions. It is
    not tied to one magazine template: callers provide plausible boxes, and the
    scorer picks the first box that both fits the translated Chinese and avoids
    already drawn elements.
    """
    unique: List[List[int]] = []
    for box in [preferred] + candidates:
        if box and box not in unique:
            unique.append(box)

    best_box = preferred
    best_score = float("inf")
    for box in unique:
        measure = _measure_text_candidate(text, box, role, max_size, min_size, line_gap_ratio)
        overlap_score = 0.0
        for occ in occupied:
            overlap_score += _overlap_ratio_rect(box, occ) * 800.0
            overlap_score += _intersection_area(box, occ) / max(1, _rect_area(box)) * 200.0
        distance = (abs(box[0] - preferred[0]) + abs(box[1] - preferred[1])) / 10000.0
        area_bonus = min(0.25, _rect_area(box) / max(1, _rect_area(preferred)) * 0.02)
        score = (
            measure["hidden_lines"] * 1000.0
            + measure["overflow_px"] * 4.0
            + overlap_score
            + (max_size - measure["font_size"]) * 2.0
            + distance
            - area_bonus
        )
        if score < best_score:
            best_score = score
            best_box = box
    return best_box


def _rect_area(rect: List[int]) -> int:
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def _intersection_area(a: List[int], b: List[int]) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _overlap_ratio_rect(a: List[int], b: List[int]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1, min(_rect_area(a), _rect_area(b)))


def choose_low_overlap_box(preferred: List[int], candidates: List[List[int]], occupied: List[List[int]]) -> List[int]:
    """
    Choose a placement box by scoring overlap with already drawn elements.

    The renderer uses this before drawing sidebars, captions, and quotes. It is a
    deterministic alternative to a visual LLM critic: generate several plausible
    boxes, measure whether they collide with existing text, and pick the box with
    the largest clean area.
    """
    all_candidates = [preferred] + [c for c in candidates if c != preferred]
    best_box = preferred
    best_score = float("inf")
    for box in all_candidates:
        score = 0.0
        for occ in occupied:
            score += _overlap_ratio_rect(box, occ) * 100.0
            score += _intersection_area(box, occ) / max(1, _rect_area(box))
        # Prefer boxes closer to the intended location when overlap is equal.
        score += abs(box[0] - preferred[0]) / 10000.0 + abs(box[1] - preferred[1]) / 10000.0
        if score < best_score:
            best_score = score
            best_box = box
    return best_box


def _register_rects(rects: List[List[int]], occupied: List[List[int]]) -> None:
    for rect in rects:
        if _rect_area(rect) > 0:
            occupied.append(rect)


def _split_author_text(text: str) -> Tuple[str, str]:
    """Split an author translation into display name and biography."""
    text = re.sub(r"\s+", " ", str(text or "")).strip(" ，,;；")
    if not text:
        return "", ""

    explicit = [p.strip() for p in re.split(r"[\n；;]+", text) if p.strip()]
    if len(explicit) >= 2:
        return explicit[0], "；".join(explicit[1:])

    match = re.match(r"^(.{2,24}?[（(][^）)]{2,80}[）)])\s*[，,、：:]?\s*(.*)$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    for sep in ["，", ",", "：", ":"]:
        if sep in text:
            head, tail = text.split(sep, 1)
            if 2 <= len(head) <= 28:
                return head.strip(), tail.strip()

    # Fallback for translations such as "Daniel Robinson is a professor..." or
    # author translations where punctuation was omitted.
    markers = ["教授", "副院长", "研究", "澳大利亚", "新西兰", "坎特伯雷", "新南威尔士"]
    cut_positions = [text.find(m) for m in markers if text.find(m) > 2]
    if cut_positions:
        cut = min(cut_positions)
        return text[:cut].strip(), text[cut:].strip()

    if len(text) > 18:
        return text[:18].strip(), text[18:].strip()
    return text, ""


def _draw_wrapped_text_at_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: List[int],
    role: str,
    size: int,
    fill: Tuple[int, int, int],
    align: str,
    line_gap_ratio: float,
    block: dict,
    page_index: int,
    drawn_role: str,
) -> List[DrawnBox]:
    font = get_font(role, size)
    gap = max(1, int(size * line_gap_ratio))
    lines = wrap_text(text, font, max(20, box[2] - box[0]))
    y = box[1]
    lh = line_height(font)
    drawn: List[DrawnBox] = []
    for line in lines:
        if y + lh > box[3]:
            break
        lw = text_width(font, line)
        if align == "right":
            x = max(box[0], box[2] - lw)
        elif align == "center":
            x = box[0] + max(0, (box[2] - box[0] - lw) // 2)
        else:
            x = box[0]
        draw.text((x, y), line, font=font, fill=fill)
        rect = [x, y, min(box[2], x + lw), y + lh]
        drawn.append(_make_drawn_box(page_index, rect, drawn_role, size, block, line))
        y += lh + gap
    return drawn


def render_author_blocks(
    draw: ImageDraw.ImageDraw,
    author_blocks: List[dict],
    template: Dict[str, Any],
    img_w: int,
    img_h: int,
    page_index: int,
    occupied: List[List[int]],
) -> List[DrawnBox]:
    """
    Render author sidebars as vertical multi-line cards.

    v5 fixes the previous long-line failure by never drawing an author name or bio
    as a single unbounded line. Each card is wrapped inside a narrow sidebar box,
    and candidate boxes are tested against already drawn title/subtitle boxes.
    """
    drawn: List[DrawnBox] = []
    if not author_blocks:
        return drawn

    base_box = clamp_box(template.get("authors"), img_w, img_h)
    if not base_box:
        base_box = [int(img_w * 0.760), int(img_h * 0.030), int(img_w * 0.945), int(img_h * 0.270)]

    sidebar_x1, sidebar_x2 = base_box[0], base_box[2]
    card_gap = int(img_h * 0.028)
    card_h = max(int(img_h * 0.105), (base_box[3] - base_box[1] - card_gap) // max(1, len(author_blocks)))

    for idx, block in enumerate(author_blocks):
        preferred = [sidebar_x1, base_box[1] + idx * (card_h + card_gap), sidebar_x2, base_box[1] + idx * (card_h + card_gap) + card_h]
        candidates = [
            preferred,
            [sidebar_x1, int(img_h * 0.030) + idx * (card_h + card_gap), sidebar_x2, int(img_h * 0.030) + idx * (card_h + card_gap) + card_h],
            [sidebar_x1, int(img_h * 0.120) + idx * (card_h + card_gap), sidebar_x2, int(img_h * 0.120) + idx * (card_h + card_gap) + card_h],
            [int(img_w * 0.715), int(img_h * 0.030) + idx * (card_h + card_gap), int(img_w * 0.945), int(img_h * 0.030) + idx * (card_h + card_gap) + card_h],
        ]
        box = choose_low_overlap_box(preferred, candidates, occupied)
        box = clamp_box(box, img_w, img_h)
        if not box:
            continue

        name, bio = _split_author_text(block.get("target_text", ""))
        if not name and not bio:
            continue

        name_size = max(16, int(img_h * 0.0125))
        bio_size = max(15, int(img_h * 0.0115))
        name_font = get_font("author", name_size)
        name_lh = line_height(name_font)
        name_box = [box[0], box[1], box[2], min(box[3], box[1] + name_lh * 3 + 4)]
        name_drawn = _draw_wrapped_text_at_size(
            draw, name, name_box, "author", name_size, COLOR_BODY, "right", 0.18, block, page_index, "author"
        )
        drawn.extend(name_drawn)
        _register_rects([d.rect for d in name_drawn], occupied)

        divider_y = name_box[1]
        if name_drawn:
            divider_y = max(d.rect[3] for d in name_drawn) + 5
        if divider_y < box[3] - 10:
            draw.line((box[0], divider_y, box[2], divider_y), fill=COLOR_ACCENT_GREEN, width=1)
            divider_rect = [box[0], divider_y, box[2], divider_y + 1]
            occupied.append(divider_rect)
            drawn.append(_make_drawn_box(page_index, divider_rect, "author_rule", 1, block, ""))

        bio_box = [box[0], divider_y + 7, box[2], box[3]]
        if bio_box[3] - bio_box[1] > 8 and bio:
            bio_drawn = _draw_wrapped_text_at_size(
                draw, bio, bio_box, "caption", bio_size, COLOR_ACCENT_GREEN, "right", 0.24, block, page_index, "author"
            )
            drawn.extend(bio_drawn)
            _register_rects([d.rect for d in bio_drawn], occupied)

    return drawn


def _footer_source_lines(state: TranslationState, img_h: int) -> List[Tuple[int, int, str]]:
    """Collect bottom running-footer candidates from PDF text and OCR.

    Copyright/photo-credit text often appears near the top or near a photo edge;
    it should not be treated as a page footer. This collector therefore focuses
    on the bottom band and requires a page-like number or a known running-footer
    marker.
    """
    items: List[Tuple[int, int, str]] = []
    for source_name in ("pdf_text_lines", "raw_ocr"):
        for line in state.get(source_name, []) or []:
            box = line.get("box") or [0, 0, 0, 0]
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            y_center = (box[1] + box[3]) / 2
            is_bottom = y_center > img_h * 0.900
            has_page_number = bool(re.search(r"\b\d{1,3}\b", text))
            has_running_marker = bool(re.search(r"Courier|UNESCO|issue|January|March|\|", text, re.I))
            if is_bottom and (has_page_number or has_running_marker):
                items.append((int(box[1]), int(box[0]), text))
    items.sort()
    return items


def _extract_source_footer_text(state: TranslationState, img_h: int) -> str:
    items = _footer_source_lines(state, img_h)
    footer = " ".join(t for _, _, t in items)
    return re.sub(r"\s+", " ", footer).strip()


def _looks_like_page_footer(text: str) -> bool:
    """Distinguish page furniture from photo credits or ordinary captions."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return False
    if "©" in text or "copyright" in text.lower() or "robertharding" in text.lower():
        return False
    return bool(re.search(r"\b\d{1,3}\b", text)) or any(
        marker in text for marker in ["信使", "Courier", "UNESCO", "联合国教科文组织", "｜", "|"]
    )


def _translate_footer_text(source: str, title_text: str, page_index: int) -> str:
    """Translate common running footer strings without another model call."""
    source = re.sub(r"\s+", " ", str(source or "")).strip()
    page_match = re.search(r"\b(\d{1,3})\b", source)
    page_no = page_match.group(1) if page_match else str(page_index + 1)
    title = re.sub(r"\s+", "", title_text or "")
    if not title:
        title = "文章标题"
    if len(title) > 14:
        title = title[:14]

    if re.search(r"UNESCO\s+Courier", source, re.I):
        issue = "2026年1—3月" if re.search(r"January\s*[-–]\s*March\s+2026", source, re.I) else ""
        suffix = f" · {issue}" if issue else ""
        return f"{page_no}｜联合国教科文组织《信使》{suffix}"
    if page_index == 0 and not source:
        return f"{page_no}｜联合国教科文组织《信使》"
    return f"{title}｜{page_no}"


def render_footer(
    draw: ImageDraw.ImageDraw,
    state: TranslationState,
    blocks: List[dict],
    template: Dict[str, Any],
    img_w: int,
    img_h: int,
    page_index: int,
    occupied: List[List[int]],
) -> List[DrawnBox]:
    drawn: List[DrawnBox] = []
    box = clamp_box(template.get("footer"), img_w, img_h)
    if not box:
        return drawn

    title_text = ""
    for b in blocks:
        if b.get("style") == "title" and b.get("target_text"):
            title_text = str(b.get("target_text"))
            break

    footer_blocks = [b for b in blocks if b.get("style") == "footer" and _looks_like_page_footer(str(b.get("target_text", "")))]
    if footer_blocks:
        footer_text = str(footer_blocks[0].get("target_text", "")).strip()
        source_block = footer_blocks[0]
    else:
        source_footer = _extract_source_footer_text(state, img_h)
        footer_text = _translate_footer_text(source_footer, title_text, page_index)
        source_block = {"id": f"footer_{page_index}", "page_num": page_index, "global_key": f"p{page_index}_footer"}

    font_size = max(13, int(img_h * 0.0105))
    font = get_font("footer", font_size)
    lh = line_height(font)
    y = box[1]
    align = "left" if page_index % 2 == 0 else "right"
    width = box[2] - box[0]
    lines = wrap_text(footer_text, font, width)
    for line in lines[:2]:
        lw = text_width(font, line)
        x = box[0] if align == "left" else box[2] - lw
        draw.text((x, y), line, font=font, fill=COLOR_BODY)
        rect = [x, y, x + lw, y + lh]
        drawn.append(_make_drawn_box(page_index, rect, "footer", font_size, source_block, line))
        occupied.append(rect)
        y += lh + 2
    return drawn


def render_non_body_elements(
    draw: ImageDraw.ImageDraw,
    blocks: List[dict],
    state: TranslationState,
    img_w: int,
    img_h: int,
    page_index: int,
    layout_suggestions: Optional[Dict[str, Any]] = None,
) -> List[DrawnBox]:
    """Render non-body elements with collision-aware placement."""
    layout_suggestions = layout_suggestions or {}
    drawn: List[DrawnBox] = []
    occupied: List[List[int]] = []
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
        text = str(block.get("target_text", "")).strip()
        if not text:
            continue
        font_size = max(18, int(img_h * 0.0115))
        font = get_font("bold", font_size)
        pad_x = max(8, int(font_size * 0.35))
        pad_y = max(3, int(font_size * 0.18))
        tw = text_width(font, text)
        th = line_height(font)
        pill_w = min(max(tw + pad_x * 2, int(img_w * 0.070)), max(20, box[2] - box[0]))
        pill_h = th + pad_y * 2
        rect = [box[0], box[1], box[0] + pill_w, box[1] + pill_h]
        draw.rectangle(rect, fill=COLOR_ACCENT_GREEN)
        draw.text((rect[0] + pad_x, rect[1] + pad_y), text, font=font, fill=(255, 255, 255))
        tri_x = rect[2] + max(4, int(font_size * 0.16))
        tri_y = rect[1] + max(2, int(pill_h * 0.18))
        draw.polygon(
            [(tri_x, tri_y), (tri_x + int(font_size * 0.45), tri_y + int(font_size * 0.32)), (tri_x, tri_y + int(font_size * 0.64))],
            fill=COLOR_ACCENT_GREEN,
        )
        drawn_rect = [rect[0], rect[1], tri_x + int(font_size * 0.45), rect[3]]
        drawn.append(_make_drawn_box(page_index, drawn_rect, "kicker", font_size, block, text))
        occupied.append(drawn_rect)

    for block in by_style.get("title", []):
        preferred = clamp_box(block.get("target_box") or template.get("title"), img_w, img_h)
        if not preferred:
            continue
        candidates = [
            preferred,
            [int(img_w * 0.075), int(img_h * 0.075), int(img_w * 0.705), int(img_h * 0.210)],
            [int(img_w * 0.075), int(img_h * 0.090), int(img_w * 0.705), int(img_h * 0.235)],
        ]
        box = choose_low_overlap_box(preferred, candidates, occupied)
        rects, font_size, truncated = draw_fixed_text(
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
        _register_rects(rects, occupied)

    for block in by_style.get("subtitle", []):
        preferred = clamp_box(block.get("target_box") or template.get("subtitle"), img_w, img_h)
        if not preferred:
            continue
        x1, y1, x2, y2 = preferred
        candidates = [
            preferred,
            clamp_box([x1, y1, x2, y2 + int(img_h * 0.055)], img_w, img_h),
            clamp_box([x1, y1, min(img_w, x2 + int(img_w * 0.080)), y2 + int(img_h * 0.070)], img_w, img_h),
            clamp_box([int(img_w * 0.075), int(img_h * 0.235), int(img_w * 0.735), int(img_h * 0.385)], img_w, img_h),
        ]
        candidates = [c for c in candidates if c]
        max_size = int(img_h * 0.022)
        min_size = int(img_h * 0.0145)
        box = choose_flexible_text_box(block.get("target_text", ""), preferred, candidates, occupied, "subtitle", max_size, min_size, 0.38)
        rects, font_size, truncated = draw_fixed_text(
            draw,
            block.get("target_text", ""),
            box,
            "subtitle",
            max_size,
            min_size,
            COLOR_BODY,
            align="left",
            line_gap_ratio=0.38,
        )
        role = "subtitle_truncated" if truncated else "subtitle"
        drawn.extend(_make_drawn_box(page_index, rect, role, font_size, block, block.get("target_text", "")) for rect in rects)
        _register_rects(rects, occupied)

    drawn.extend(render_author_blocks(draw, by_style.get("author", []), template, img_w, img_h, page_index, occupied))

    for block in by_style.get("caption", []):
        preferred = clamp_box(block.get("target_box") or template.get("caption"), img_w, img_h)
        if not preferred:
            continue
        x1, y1, x2, y2 = preferred
        candidates = [
            preferred,
            clamp_box([x1, y1, x2, y2 + int(img_h * 0.025)], img_w, img_h),
            clamp_box([int(img_w * 0.700), int(img_h * 0.045), int(img_w * 0.940), int(img_h * 0.125)], img_w, img_h),
        ]
        candidates = [c for c in candidates if c]
        max_size = int(img_h * 0.0125)
        min_size = int(img_h * 0.0090)
        box = choose_flexible_text_box(block.get("target_text", ""), preferred, candidates, occupied, "caption", max_size, min_size, 0.18)
        rects, font_size, truncated = draw_fixed_text(
            draw,
            block.get("target_text", ""),
            box,
            "caption",
            max_size,
            min_size,
            COLOR_BODY,
            align="left",
            line_gap_ratio=0.18,
        )
        role = "caption_truncated" if truncated else "caption"
        drawn.extend(_make_drawn_box(page_index, rect, role, font_size, block, block.get("target_text", "")) for rect in rects)
        _register_rects(rects, occupied)

    for block in by_style.get("quote", []):
        text = str(block.get("target_text", "")).strip()
        if not text:
            continue
        preferred = clamp_box(block.get("target_box") or template.get("quote"), img_w, img_h)
        if not preferred:
            continue
        x1, y1, x2, y2 = preferred
        candidates = [
            preferred,
            clamp_box([x1, y1, x2, y2 + int(img_h * 0.045)], img_w, img_h),
        ]
        if page_index == 0:
            candidates.extend([
                clamp_box([int(img_w * 0.385), int(img_h * 0.775), int(img_w * 0.920), int(img_h * 0.940)], img_w, img_h),
                clamp_box([int(img_w * 0.360), int(img_h * 0.740), int(img_w * 0.925), int(img_h * 0.940)], img_w, img_h),
            ])
        else:
            candidates.extend([
                clamp_box([int(img_w * 0.705), int(img_h * 0.140), int(img_w * 0.930), int(img_h * 0.400)], img_w, img_h),
                clamp_box([int(img_w * 0.690), int(img_h * 0.120), int(img_w * 0.940), int(img_h * 0.420)], img_w, img_h),
            ])
        candidates = [c for c in candidates if c]
        max_size = int(img_h * 0.034)
        min_size = int(img_h * 0.021)
        box = choose_flexible_text_box(text, preferred, candidates, occupied, "quote", max_size, min_size, 0.28)
        rects, font_size, truncated = draw_fixed_text(
            draw,
            text,
            box,
            "quote",
            max_size,
            min_size,
            COLOR_ACCENT_GREEN,
            align="left",
            line_gap_ratio=0.28,
        )
        if rects:
            mark_size = int(img_h * 0.050)
            mark_rect = [box[0], max(0, box[1] - int(img_h * 0.025)), box[0] + mark_size, box[1] + int(mark_size * 0.65)]
            draw.text((mark_rect[0], mark_rect[1]), "“", font=get_font("quote", mark_size), fill=COLOR_QUOTE_MARK)
            occupied.append(mark_rect)
            drawn.append(_make_drawn_box(page_index, mark_rect, "quote_mark", mark_size, block, "“"))
        role = "quote_truncated" if truncated else "quote"
        drawn.extend(_make_drawn_box(page_index, rect, role, font_size, block, text) for rect in rects)
        _register_rects(rects, occupied)

    drawn.extend(render_footer(draw, state, blocks, template, img_w, img_h, page_index, occupied))
    return drawn

def get_body_frames_for_page(page_index: int, img_w: int, img_h: int, has_title: bool) -> List[FlowFrame]:
    template = infer_chinese_template(img_w, img_h, page_index, has_title)
    frames: List[FlowFrame] = []
    for box in template.get("body_columns", []):  # type: ignore[union-attr]
        clamped = clamp_box(box, img_w, img_h)
        if clamped:
            frames.append(FlowFrame(page_index=page_index, box=clamped, role="body"))
    return frames


def expected_non_body_refs_from_states(page_states: List[TranslationState]) -> List[str]:
    """Return refs for page furniture that should produce visible text.

    These refs allow the layout critic to flag dropped pull quotes, captions,
    kickers, titles, author sidebars, and footers.
    """
    refs: List[str] = []
    non_body_styles = {"kicker", "title", "subtitle", "author", "quote", "caption", "footer"}
    for state in page_states:
        page_num = int(state.get("page_num", 0))
        for block in state.get("translated_blocks", []) or []:
            if block.get("style") in non_body_styles and str(block.get("target_text", "")).strip():
                refs.append(str(block.get("global_key") or f"p{page_num}_b{block.get('id')}"))
    return refs




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
        non_body_drawn.extend(render_non_body_elements(draw_by_page[page_index], blocks, state, img_w, img_h, page_index, layout_suggestions))
        body_frames.extend(get_body_frames_for_page(page_index, img_w, img_h, has_title))

    img_w, img_h = images[0].size
    atoms = build_body_atoms(page_states, document_translation)
    refs = expected_body_refs(page_states)
    non_body_refs = expected_non_body_refs_from_states(page_states)

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
        expected_non_body_refs=non_body_refs,
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
