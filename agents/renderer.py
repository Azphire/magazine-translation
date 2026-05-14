from __future__ import annotations

import gc
import io
import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont

from config import (
    BODY_FIRST_LINE_INDENT_EM,
    BODY_FONT_SIZE,
    BODY_HEADING_GAP_RATIO,
    BODY_LINE_GAP_RATIO,
    BODY_PARAGRAPH_GAP_RATIO,
    COLOR_ACCENT_GREEN,
    COLOR_BODY,
    COLOR_MUTED,
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
from utils.layout_utils import clamp_box, infer_chinese_template, union_boxes
from utils.logger import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTHONMALLOC"] = "malloc"

logger.info("[Renderer] Initialized v3: PDF/OCR/CV erasing + document-level cross-page body flow.")

_FONT_BYTES_CACHE: Dict[str, bytes] = {}
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font_path(role: str) -> str:
    candidates = {
        "title": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "subtitle": [FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "author": [FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "body": [FONT_SERIF, FONT_REGULAR, FONT_FALLBACK],
        "body_heading": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "quote": [FONT_BOLD, FONT_MEDIUM, FONT_REGULAR, FONT_FALLBACK],
        "caption": [FONT_REGULAR, FONT_FALLBACK],
        "footer": [FONT_REGULAR, FONT_FALLBACK],
    }
    for p in candidates.get(role, [FONT_REGULAR, FONT_FALLBACK]):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("No usable CJK font found. Put fonts into data/fonts or provide data/simhei.ttf.")


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


def text_bbox(font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int, int, int]:
    return font.getbbox(text or " ")


def text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    b = text_bbox(font, text)
    return b[2] - b[0]


def text_height(font: ImageFont.FreeTypeFont, text: str = "国") -> int:
    b = text_bbox(font, text)
    return b[3] - b[1]


def _box_pad(box: List[int]) -> int:
    h = max(1, box[3] - box[1])
    return max(ERASE_MIN_PAD, min(ERASE_MAX_PAD, int(h * ERASE_BOX_PAD_RATIO)))




def _iou(a: List[int], b: List[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, area_a + area_b - inter)


def detect_cv_text_boxes(raw_img: Image.Image, existing_boxes: List[List[int]], basename: str) -> List[List[int]]:
    """
    Fallback eraser for non-selectable or OCR-missed English.
    It detects dark/colored glyph strokes on white paper background and groups them into line boxes.
    This is intentionally used only for erasing, not for translation.
    """
    if not CV_TEXT_DETECT_ENABLE:
        return []

    arr = np.array(raw_img.convert("RGB"))
    img_h, img_w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # White-context gate: text printed on magazine paper has abundant white nearby; photo regions do not.
    white_pixels = (gray > 238).astype(np.uint8)
    k = max(21, int(min(img_w, img_h) * 0.018))
    if k % 2 == 0:
        k += 1
    white_context = cv2.blur(white_pixels.astype(np.float32), (k, k)) > float(CV_WHITE_CONTEXT_THRESHOLD)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturated_colored = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 245)
    darkish = gray < int(CV_DARK_GRAY_THRESHOLD)
    candidate = ((darkish | saturated_colored) & white_context).astype(np.uint8) * 255

    # Remove very light paper noise; group glyphs horizontally into text-line boxes.
    candidate = cv2.medianBlur(candidate, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(CV_LINE_DILATE_W), int(CV_LINE_DILATE_H)))
    line_mask = cv2.dilate(candidate, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    boxes: List[List[int]] = []
    max_h = max(8, int(img_h * CV_COMPONENT_MAX_HEIGHT_RATIO))
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < CV_COMPONENT_MIN_AREA:
            continue
        if w < CV_COMPONENT_MIN_WIDTH or h < 3 or h > max_h:
            continue
        # Avoid selecting broad page/photo edges.
        if w > img_w * 0.78 or h > img_h * 0.10:
            continue
        box = [int(x), int(y), int(x + w), int(y + h)]
        # Add a tiny crop sanity check: at least a few dark pixels should be inside.
        crop = candidate[y:y + h, x:x + w]
        if crop.size == 0 or int(np.count_nonzero(crop)) < 3:
            continue
        if any(_iou(box, eb) > 0.72 for eb in existing_boxes):
            continue
        boxes.append(box)

    # Merge boxes that overlap after line dilation.
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = []
    for b in boxes:
        if merged and not (b[0] > merged[-1][2] + 20 or b[1] > merged[-1][3] + 8):
            m = merged[-1]
            merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
        else:
            merged.append(b)

    if DEBUG_SAVE_INTERMEDIATE:
        os.makedirs(TEMP_DIR, exist_ok=True)
        Image.fromarray(candidate).save(os.path.join(TEMP_DIR, f"{basename}_cv_text_candidates.png"))
        Image.fromarray(line_mask).save(os.path.join(TEMP_DIR, f"{basename}_cv_text_lines.png"))

    logger.info(f"[Renderer] CV fallback detected {len(merged)} extra text-line boxes for erasing.")
    return merged

def collect_erase_boxes(state: TranslationState, img_w: int, img_h: int) -> List[List[int]]:
    """
    Use every available text coordinate source, not just parser blocks.
    This is the core fix for English residue.
    Priority: PDF-native text boxes -> parser erase_boxes -> OCR fallback.
    """
    boxes: List[List[int]] = []

    for src_name in ("pdf_text_lines", "raw_ocr", "cv_text_lines"):
        for line in state.get(src_name, []) or []:
            box = line.get("box")
            cb = clamp_box(box, img_w, img_h) if box else None
            if cb:
                boxes.append(cb)

    for b in state.get("translated_blocks", []) or []:
        for box in b.get("erase_boxes") or []:
            cb = clamp_box(box, img_w, img_h)
            if cb:
                boxes.append(cb)
        for key in ("source_box", "box"):
            box = b.get(key)
            cb = clamp_box(box, img_w, img_h) if box else None
            if cb:
                boxes.append(cb)

    # De-duplicate near identical boxes.
    seen = set()
    unique = []
    for b in boxes:
        q = tuple(int(v // 3) for v in b)
        if q not in seen:
            seen.add(q)
            unique.append(b)
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
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Close tiny gaps between letters/words and guarantee anti-aliased halos are included.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    inpainted = cv2.inpaint(arr, mask, ERASE_INPAINT_RADIUS, cv2.INPAINT_TELEA)

    # On white page regions, inpainting may leave faint texture. Force uniform white where the original box background is already near-white.
    for box in erase_boxes:
        pad = _box_pad(box)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
        roi = arr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        med = np.median(roi.reshape(-1, 3), axis=0)
        # Most magazine text is printed on white page background. This hard fill removes residue completely.
        if float(np.mean(med)) > 222 and float(np.std(roi.reshape(-1, 3))) < 58:
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


FORBIDDEN_LINE_START = set("，。；：！？、）】》〉」』”’%")
FORBIDDEN_LINE_END = set("（【《〈「『“‘")


def tokenize_cjk_mixed(text: str) -> List[str]:
    tokens: List[str] = []
    buf = ""
    for ch in text:
        if ch.isspace():
            if buf:
                tokens.append(buf)
                buf = ""
            continue
        if ch.isascii() and (ch.isalnum() or ch in ".,-/()"):
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
    if buf:
        tokens.append(buf)
    return tokens


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int, indent_w: int = 0) -> List[str]:
    text = re.sub(r"\s+", "", text.strip())
    if not text:
        return []
    tokens = tokenize_cjk_mixed(text)
    lines: List[str] = []
    cur = ""
    cur_limit = max_w - indent_w
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        test = cur + tok
        if text_width(font, test) <= max(20, cur_limit):
            cur = test
            i += 1
        else:
            if not cur:
                # Extremely long Latin token; force split.
                cur = tok
                i += 1
            # Avoid bad punctuation at line start by pulling punctuation back.
            if i < len(tokens) and tokens[i] in FORBIDDEN_LINE_START:
                cur += tokens[i]
                i += 1
            while cur and cur[-1] in FORBIDDEN_LINE_END and i < len(tokens):
                cur += tokens[i]
                i += 1
            lines.append(cur)
            cur = ""
            cur_limit = max_w
    if cur:
        lines.append(cur)
    return lines


def draw_lines(draw: ImageDraw.ImageDraw, lines: List[str], font: ImageFont.FreeTypeFont, box: List[int], fill, line_gap: int, align: str = "left", start_y: int | None = None) -> int:
    x1, y1, x2, y2 = box
    y = y1 if start_y is None else start_y
    lh = text_height(font)
    for line in lines:
        if y + lh > y2:
            break
        lw = text_width(font, line)
        if align == "center":
            x = x1 + max(0, (x2 - x1 - lw) // 2)
        elif align == "right":
            x = x2 - lw
        else:
            x = x1
        draw.text((x, y), line, font=font, fill=fill)
        y += lh + line_gap
    return y


def fit_text_to_box(text: str, box: List[int], role: str, max_size: int, min_size: int, line_gap_ratio: float = 0.25) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    w = max(5, box[2] - box[0])
    h = max(5, box[3] - box[1])
    for size in range(max_size, min_size - 1, -1):
        font = get_font(role, size)
        line_gap = max(1, int(size * line_gap_ratio))
        lines = wrap_text(text, font, w)
        total_h = len(lines) * text_height(font) + max(0, len(lines) - 1) * line_gap
        if total_h <= h:
            return font, lines, line_gap
    font = get_font(role, min_size)
    return font, wrap_text(text, font, w), max(1, int(min_size * line_gap_ratio))


def color_for(role: str):
    return {
        "title_red": COLOR_TITLE_RED,
        "subhead_red": COLOR_SUBHEAD_RED,
        "accent_green": COLOR_ACCENT_GREEN,
        "muted": COLOR_MUTED,
        "body": COLOR_BODY,
        "body_black": COLOR_BODY,
    }.get(role, COLOR_BODY)


def _sorted_blocks(state: TranslationState) -> List[dict]:
    return sorted(state.get("translated_blocks", []) or [], key=lambda b: (int(b.get("reading_order", b.get("id", 0))), int(b.get("id", 0))))


def render_non_body(draw: ImageDraw.ImageDraw, blocks: List[dict], img_w: int, img_h: int, page_num: int) -> None:
    has_title = any(b.get("style") == "title" for b in blocks)
    tpl = infer_chinese_template(img_w, img_h, page_num, has_title)

    by_style = {s: [b for b in blocks if b.get("style") == s] for s in ["kicker", "title", "subtitle", "author", "quote", "caption", "footer"]}

    for b in by_style.get("kicker", []):
        box = clamp_box(b.get("target_box") or tpl.get("kicker"), img_w, img_h)
        if not box:
            continue
        font = get_font("caption", max(10, int(img_w * 0.010)))
        draw.text((box[0], box[1]), b.get("target_text", ""), font=font, fill=COLOR_MUTED)
        # Small green arrow/dot like magazine section marker.
        x = box[0] + text_width(font, b.get("target_text", "")) + 5
        y = box[1] + 3
        draw.polygon([(x, y), (x + 12, y + 8), (x, y + 16)], fill=COLOR_ACCENT_GREEN)

    for b in by_style.get("title", []):
        box = clamp_box(b.get("target_box") or tpl.get("title"), img_w, img_h)
        if not box:
            continue
        font, lines, gap = fit_text_to_box(b.get("target_text", ""), box, "title", int(img_w * 0.060), int(img_w * 0.032), 0.08)
        draw_lines(draw, lines, font, box, color_for(b.get("color_role", "title_red")), gap, "left")

    for b in by_style.get("subtitle", []):
        box = clamp_box(b.get("target_box") or tpl.get("subtitle"), img_w, img_h)
        if not box:
            continue
        font, lines, gap = fit_text_to_box(b.get("target_text", ""), box, "subtitle", int(img_w * 0.030), int(img_w * 0.018), 0.26)
        draw_lines(draw, lines, font, box, COLOR_BODY, gap, "left")

    if by_style.get("author"):
        box = clamp_box(tpl.get("authors") or by_style["author"][0].get("target_box"), img_w, img_h)
        if box:
            y = box[1]
            name_font = get_font("author", int(img_w * 0.011))
            bio_font = get_font("caption", int(img_w * 0.010))
            for b in by_style["author"]:
                text = b.get("target_text", "")
                parts = [p.strip() for p in re.split(r"[\n；;]+", text) if p.strip()]
                if not parts:
                    continue
                name = parts[0]
                bio = "；".join(parts[1:]) if len(parts) > 1 else ""
                lw = text_width(name_font, name)
                draw.text((box[2] - lw, y), name, font=name_font, fill=COLOR_BODY)
                y += text_height(name_font) + 5
                draw.line((box[0], y, box[2], y), fill=COLOR_ACCENT_GREEN, width=1)
                y += 7
                for line in wrap_text(bio, bio_font, box[2] - box[0]):
                    lw = text_width(bio_font, line)
                    draw.text((box[2] - lw, y), line, font=bio_font, fill=COLOR_ACCENT_GREEN)
                    y += text_height(bio_font) + 3
                y += int(img_h * 0.035)

    for b in by_style.get("caption", []):
        # Render only small captions outside photo. Body-like big captions should stay out of main flow.
        box = clamp_box(b.get("target_box") or tpl.get("caption"), img_w, img_h)
        if not box:
            continue
        font, lines, gap = fit_text_to_box(b.get("target_text", ""), box, "caption", int(img_w * 0.013), int(img_w * 0.009), 0.15)
        draw_lines(draw, lines, font, box, COLOR_BODY, gap, "left")

    for b in by_style.get("quote", []):
        box = clamp_box(b.get("target_box") or tpl.get("quote"), img_w, img_h)
        if not box:
            continue
        quote_font = get_font("quote", int(img_w * 0.035))
        draw.text((box[0], box[1] - int(img_w * 0.025)), "“", font=get_font("quote", int(img_w * 0.060)), fill=(198, 106, 82))
        font, lines, gap = fit_text_to_box(b.get("target_text", ""), box, "quote", int(img_w * 0.034), int(img_w * 0.020), 0.20)
        draw_lines(draw, lines, font, box, COLOR_ACCENT_GREEN, gap, "left")


def make_body_elements(blocks: List[dict]) -> List[dict]:
    elements: List[dict] = []
    for b in blocks:
        style = b.get("style")
        if style == "body_heading":
            elements.append({"type": "heading", "text": b.get("target_text", "")})
        elif style == "body":
            # Split paragraphs but do not split every OCR line; this is the fix for fragmented variable-size paragraphs.
            text = b.get("target_text", "").strip()
            if text:
                for para in re.split(r"\n{2,}|(?<=。)\s*\n", text):
                    para = para.strip()
                    if para:
                        elements.append({"type": "paragraph", "text": para})
    return elements


def render_body_flow(draw: ImageDraw.ImageDraw, elements: List[dict], columns: List[List[int]], flow_context: Dict, img_w: int, page_num: int) -> Dict:
    """
    Flow all body paragraphs through all columns with a single font size.
    Overflow is carried to the next page through flow_context['body_overflow_elements'].
    """
    existing_overflow = flow_context.pop("body_overflow_elements", []) or []
    queue = existing_overflow + elements
    if not queue:
        return flow_context

    if flow_context.get("body_font_size"):
        body_size = int(flow_context["body_font_size"])
    else:
        body_size = int(BODY_FONT_SIZE) if BODY_FONT_SIZE else max(16, int(img_w * 0.0138))
        flow_context["body_font_size"] = body_size

    body_font = get_font("body", body_size)
    heading_font = get_font("body_heading", int(body_size * 1.28))
    line_gap = max(3, int(body_size * BODY_LINE_GAP_RATIO))
    para_gap = max(4, int(body_size * BODY_PARAGRAPH_GAP_RATIO))
    heading_gap = max(4, int(body_size * BODY_HEADING_GAP_RATIO))
    indent_w = text_width(body_font, "中") * BODY_FIRST_LINE_INDENT_EM

    col_idx = 0
    y = columns[0][1] if columns else 0
    remaining: List[dict] = []

    def advance_column() -> bool:
        nonlocal col_idx, y
        col_idx += 1
        if col_idx >= len(columns):
            return False
        y = columns[col_idx][1]
        return True

    elem_idx = 0
    while elem_idx < len(queue):
        if col_idx >= len(columns):
            remaining.extend(queue[elem_idx:])
            break
        col = columns[col_idx]
        elem = queue[elem_idx]
        is_heading = elem.get("type") == "heading"
        font = heading_font if is_heading else body_font
        fill = COLOR_SUBHEAD_RED if is_heading else COLOR_BODY
        extra_after = heading_gap if is_heading else para_gap
        first_indent = 0 if is_heading else indent_w
        lines = wrap_text(elem.get("text", ""), font, col[2] - col[0], indent_w=first_indent)
        if not lines:
            elem_idx += 1
            continue

        lh = text_height(font)
        line_i = 0
        while line_i < len(lines):
            if y + lh > col[3]:
                if not advance_column():
                    # Keep the unfinished paragraph for next page.
                    rest_text = "".join(lines[line_i:])
                    if rest_text:
                        remaining.append({"type": elem.get("type", "paragraph"), "text": rest_text})
                    remaining.extend(queue[elem_idx + 1:])
                    flow_context["body_overflow_elements"] = remaining
                    logger.warning(f"[Renderer] Page {page_num}: body overflow carried to next page: {len(remaining)} elements")
                    return flow_context
                col = columns[col_idx]
                continue

            line = lines[line_i]
            x = col[0]
            if line_i == 0 and not is_heading:
                x += first_indent
            draw.text((x, y), line, font=font, fill=fill)
            y += lh + line_gap
            line_i += 1

        y += extra_after
        elem_idx += 1

    flow_context.pop("body_overflow_elements", None)
    return flow_context


def renderer_node(state: TranslationState) -> dict:
    logger.info("[Renderer] Starting v2 render process...")
    img_path = state.get("image_path")
    page_num = int(state.get("page_num", 0))
    flow_context = dict(state.get("flow_context") or {})

    if not img_path or not os.path.exists(img_path):
        return {"output_image_path": None, "debug_paths": {}, "flow_context": flow_context}

    with Image.open(img_path) as im:
        raw_img = im.convert("RGB")
    img_w, img_h = raw_img.size
    basename = os.path.splitext(os.path.basename(img_path))[0]

    erase_boxes = collect_erase_boxes(state, img_w, img_h)
    cv_extra_boxes = detect_cv_text_boxes(raw_img, erase_boxes, basename)
    if cv_extra_boxes:
        state["cv_text_lines"] = [{"id": i, "box": b, "text": "", "source": "cv"} for i, b in enumerate(cv_extra_boxes)]
        erase_boxes = collect_erase_boxes(state, img_w, img_h)
    logger.info(f"[Renderer] Erasing {len(erase_boxes)} text boxes using PDF/OCR/parser/CV coordinates.")
    clean_img, debug_paths = erase_original_text(raw_img, erase_boxes, basename)

    draw = ImageDraw.Draw(clean_img)
    blocks = _sorted_blocks(state)
    has_title = any(b.get("style") == "title" for b in blocks)
    tpl = infer_chinese_template(img_w, img_h, page_num, has_title)

    # Non-body elements are rendered in fixed magazine slots.
    render_non_body(draw, blocks, img_w, img_h, page_num)

    # Body elements are not rendered block-by-block. In v3, document-level translation can provide
    # a single global body queue that starts on page 0 and carries overflow across pages.
    if flow_context.get("use_global_body_flow"):
        if not flow_context.get("_global_body_started"):
            body_elements = list(flow_context.get("global_body_queue", []) or [])
            flow_context["_global_body_started"] = True
        else:
            body_elements = []
    else:
        body_elements = make_body_elements(blocks)
    columns = [clamp_box(c, img_w, img_h) for c in tpl.get("body_columns", [])]  # type: ignore
    columns = [c for c in columns if c]
    flow_context = render_body_flow(draw, body_elements, columns, flow_context, img_w, page_num)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"final_{basename}.jpg")
    clean_img.save(out_path, quality=95)

    logger.info(f"[Renderer] Saved: {out_path}")
    del raw_img, clean_img, draw
    gc.collect()

    return {"output_image_path": out_path, "debug_paths": debug_paths, "flow_context": flow_context}
