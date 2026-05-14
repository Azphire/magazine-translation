from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import ImageDraw, ImageFont


@dataclass
class FlowStyle:
    body_font_size: int
    body_line_gap_ratio: float
    body_para_gap_ratio: float
    heading_font_size: int
    heading_gap_ratio: float


@dataclass
class FlowFrame:
    page_index: int
    box: List[int]
    role: str = "body"


@dataclass
class FlowAtom:
    kind: str
    text: str
    source_ids: List[str] = field(default_factory=list)


@dataclass
class DrawnBox:
    page_index: int
    rect: List[int]
    role: str
    font_size: int
    source_ids: List[str]
    text: str


FORBIDDEN_LINE_START = "，。！？；：、）】》〉」』”’%"
FORBIDDEN_LINE_END = "（【《〈「『“‘"


def rect_area(rect: List[int]) -> int:
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def normalize_zh_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.replace(" ,", "，").replace(" .", "。")
    return text


def block_ref(page_num: int, block_id: Any) -> str:
    return f"p{int(page_num)}_b{block_id}"


def is_body_style(style: str) -> bool:
    return style in {"body", "body_heading", "section_heading"}


def is_heading_style(style: str) -> bool:
    return style in {"body_heading", "section_heading"}


def build_body_atoms(page_states: List[Dict[str, Any]], document_translation: Dict[str, Any]) -> List[FlowAtom]:
    """
    Build a single article body stream.

    If document-level translation produced global_body_flow, it is authoritative.
    Otherwise, the function falls back to translated page body blocks in page order.
    """
    atoms: List[FlowAtom] = []
    global_flow = document_translation.get("global_body_flow") or []
    if global_flow:
        for elem in global_flow:
            kind = "heading" if elem.get("type") == "heading" else "paragraph"
            text = normalize_zh_text(elem.get("text", ""))
            refs = [str(r) for r in elem.get("source_refs", [])]
            if text:
                atoms.append(FlowAtom(kind=kind, text=text, source_ids=refs))
        return atoms

    for state in page_states:
        page_num = int(state.get("page_num", 0))
        blocks = sorted(
            state.get("translated_blocks", []) or [],
            key=lambda b: (int(b.get("reading_order", b.get("id", 0))), int(b.get("id", 0))),
        )
        for block in blocks:
            style = block.get("style", "body")
            if not is_body_style(style):
                continue
            text = normalize_zh_text(block.get("target_text", ""))
            if not text:
                continue
            kind = "heading" if is_heading_style(style) else "paragraph"
            atoms.append(FlowAtom(kind=kind, text=text, source_ids=[block_ref(page_num, block.get("id"))]))
    return atoms


def expected_body_refs(page_states: List[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    for state in page_states:
        page_num = int(state.get("page_num", 0))
        parsed_blocks = (state.get("parsed_json") or {}).get("blocks", []) or []
        for block in parsed_blocks:
            if is_body_style(block.get("style", "body")):
                refs.append(block_ref(page_num, block.get("id")))
    return refs


def tokenize_mixed_text(text: str) -> List[str]:
    tokens: List[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            tokens.append(buf)
            buf = ""

    for ch in str(text or ""):
        if ch.isspace():
            flush()
            continue
        if ch.isascii() and (ch.isalnum() or ch in "-.()/"):
            buf += ch
        else:
            flush()
            tokens.append(ch)
    flush()
    return tokens


def text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    if not text:
        return 0
    box = font.getbbox(text)
    return box[2] - box[0]


def line_height(font: ImageFont.FreeTypeFont) -> int:
    box = font.getbbox("\u56fd")
    return max(1, box[3] - box[1])


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, first_line_indent: str = "") -> List[str]:
    tokens = tokenize_mixed_text(text)
    lines: List[str] = []
    current = first_line_indent
    current_limit = max_width

    for token in tokens:
        candidate = current + token
        if text_width(font, candidate) <= max(20, current_limit):
            current = candidate
            continue

        if current:
            if token in FORBIDDEN_LINE_START:
                current += token
                lines.append(current.rstrip())
                current = ""
            else:
                if current[-1:] in FORBIDDEN_LINE_END:
                    current += token
                    lines.append(current.rstrip())
                    current = ""
                else:
                    lines.append(current.rstrip())
                    current = token.lstrip()
        else:
            lines.append(token)
            current = ""
        current_limit = max_width

    if current.strip():
        lines.append(current.rstrip())

    fixed: List[str] = []
    for line in lines:
        if fixed and line and line[0] in FORBIDDEN_LINE_START:
            fixed[-1] += line[0]
            line = line[1:]
        if line:
            fixed.append(line)
    return fixed


def simulate_flow(
    atoms: List[FlowAtom],
    frames: List[FlowFrame],
    font_regular: ImageFont.FreeTypeFont,
    font_heading: ImageFont.FreeTypeFont,
    style: FlowStyle,
) -> Dict[str, Any]:
    frame_used = [0 for _ in frames]
    frame_chars = [0 for _ in frames]
    frame_capacity = [max(1, f.box[3] - f.box[1]) for f in frames]
    frame_idx = 0
    unrendered_chars = 0

    for atom in atoms:
        if frame_idx >= len(frames):
            unrendered_chars += len(atom.text)
            continue

        frame_width = max(20, frames[frame_idx].box[2] - frames[frame_idx].box[0])
        if atom.kind == "heading":
            lines = wrap_text(atom.text, font_heading, frame_width)
            lh = line_height(font_heading)
            needed = len(lines) * lh + int(style.heading_font_size * style.heading_gap_ratio)
            if frame_used[frame_idx] + needed > frame_capacity[frame_idx]:
                frame_idx += 1
                if frame_idx >= len(frames):
                    unrendered_chars += len(atom.text)
                    continue
            frame_used[frame_idx] += needed
            frame_chars[frame_idx] += len(atom.text)
            continue

        lines = wrap_text(atom.text, font_regular, frame_width, first_line_indent="　　")
        lh = line_height(font_regular)
        line_gap = int(style.body_font_size * style.body_line_gap_ratio)
        para_gap = int(style.body_font_size * style.body_para_gap_ratio)

        for line_index, line in enumerate(lines):
            if frame_idx >= len(frames):
                unrendered_chars += len("".join(lines[line_index:]))
                break
            if frame_used[frame_idx] + lh > frame_capacity[frame_idx]:
                frame_idx += 1
                if frame_idx >= len(frames):
                    unrendered_chars += len("".join(lines[line_index:]))
                    break
            frame_used[frame_idx] += lh + line_gap
            frame_chars[frame_idx] += len(line)
        if frame_idx < len(frames):
            frame_used[frame_idx] += para_gap

    total_used = sum(frame_used)
    total_capacity = sum(frame_capacity)
    page_stats: Dict[int, Dict[str, int]] = {}
    for i, frame in enumerate(frames):
        stats = page_stats.setdefault(frame.page_index, {"used": 0, "capacity": 0, "chars": 0})
        stats["used"] += frame_used[i]
        stats["capacity"] += frame_capacity[i]
        stats["chars"] += frame_chars[i]

    return {
        "fill_ratio": total_used / max(1, total_capacity),
        "frame_used": frame_used,
        "frame_capacity": frame_capacity,
        "page_stats": page_stats,
        "unrendered_chars": unrendered_chars,
    }


def choose_global_body_style(
    atoms: List[FlowAtom],
    frames: List[FlowFrame],
    font_loader: Callable[[str, int], ImageFont.FreeTypeFont],
    img_h: int,
    config: Dict[str, Any],
    layout_suggestions: Optional[Dict[str, Any]] = None,
) -> FlowStyle:
    """Choose one body style for the full article by simulating multiple candidates."""
    layout_suggestions = layout_suggestions or {}
    min_body = int(img_h * float(config["BODY_MIN_SIZE_RATIO"]))
    max_body = int(img_h * float(config["BODY_MAX_SIZE_RATIO"]))

    if config.get("BODY_FONT_SIZE"):
        min_body = max_body = int(config["BODY_FONT_SIZE"])

    delta = int(layout_suggestions.get("body_font_delta", 0))
    min_body = max(12, min_body + delta)
    max_body = max(min_body, max_body + delta)

    target_fill = float(config.get("BODY_TARGET_FILL_RATIO", 0.74))
    best_style: Optional[FlowStyle] = None
    best_score = 10**9

    for size in range(min_body, max_body + 1):
        for line_gap_ratio in config["BODY_LINE_GAP_CANDIDATES"]:
            adjusted_line_gap = max(0.10, min(0.90, float(line_gap_ratio) + float(layout_suggestions.get("body_line_gap_delta", 0.0))))
            for para_gap_ratio in config["BODY_PARAGRAPH_GAP_CANDIDATES"]:
                adjusted_para_gap = max(0.20, min(1.80, float(para_gap_ratio) + float(layout_suggestions.get("body_para_gap_delta", 0.0))))
                style = FlowStyle(
                    body_font_size=size,
                    body_line_gap_ratio=adjusted_line_gap,
                    body_para_gap_ratio=adjusted_para_gap,
                    heading_font_size=max(size + 2, int(size * float(config.get("BODY_HEADING_SIZE_RATIO", 1.38)))),
                    heading_gap_ratio=float(config.get("BODY_HEADING_GAP_RATIO", 0.75)),
                )
                font_regular = font_loader("body", style.body_font_size)
                font_heading = font_loader("body_heading", style.heading_font_size)
                sim = simulate_flow(atoms, frames, font_regular, font_heading, style)
                overflow_penalty = 10.0 if sim["unrendered_chars"] > 0 else 0.0
                score = abs(sim["fill_ratio"] - target_fill) + overflow_penalty

                page_fills = [s["used"] / max(1, s["capacity"]) for s in sim["page_stats"].values() if s["chars"] > 30]
                if len(page_fills) >= 2:
                    score += (max(page_fills) - min(page_fills)) * 0.55

                if score < best_score:
                    best_score = score
                    best_style = style

    if best_style is None:
        size = max(14, int(img_h * 0.012))
        best_style = FlowStyle(size, 0.36, 0.65, int(size * 1.38), 0.75)
    return best_style


def render_body_flow(
    draw_by_page: Dict[int, ImageDraw.ImageDraw],
    atoms: List[FlowAtom],
    frames: List[FlowFrame],
    font_loader: Callable[[str, int], ImageFont.FreeTypeFont],
    style: FlowStyle,
    fill: Tuple[int, int, int],
    heading_fill: Tuple[int, int, int],
) -> Tuple[List[DrawnBox], int, Dict[int, Dict[str, int]]]:
    """
    Render the continuous article body across all frames.

    v5 fixes page fill diagnostics by tracking usage per frame instead of using
    a single max-used value per page. The previous implementation underreported
    page usage when several columns were partially filled.
    """
    drawn: List[DrawnBox] = []
    if not frames:
        return drawn, sum(len(a.text) for a in atoms), {}

    frame_used = [0 for _ in frames]
    frame_chars = [0 for _ in frames]
    frame_capacity = [max(1, f.box[3] - f.box[1]) for f in frames]

    frame_idx = 0
    y_cursor = frames[0].box[1]
    unrendered_chars = 0
    font_regular = font_loader("body", style.body_font_size)
    font_heading = font_loader("body_heading", style.heading_font_size)

    def move_to_frame(idx: int) -> bool:
        nonlocal frame_idx, y_cursor
        frame_idx = idx
        if frame_idx >= len(frames):
            return False
        y_cursor = frames[frame_idx].box[1]
        return True

    def mark_usage(frame_index: int, cursor_y: int, chars: int) -> None:
        frame = frames[frame_index]
        frame_used[frame_index] = max(frame_used[frame_index], max(0, cursor_y - frame.box[1]))
        frame_chars[frame_index] += max(0, chars)

    for atom in atoms:
        if frame_idx >= len(frames):
            unrendered_chars += len(atom.text)
            continue

        frame = frames[frame_idx]
        x1, y1, x2, y2 = frame.box
        width = max(20, x2 - x1)

        if atom.kind == "heading":
            lines = wrap_text(atom.text, font_heading, width)
            lh = line_height(font_heading)
            gap_after = int(style.heading_font_size * style.heading_gap_ratio)
            needed = len(lines) * lh + gap_after
            if y_cursor + needed > y2:
                if not move_to_frame(frame_idx + 1):
                    unrendered_chars += len(atom.text)
                    continue
                frame = frames[frame_idx]
                x1, y1, x2, y2 = frame.box
                width = max(20, x2 - x1)
                lines = wrap_text(atom.text, font_heading, width)

            for line in lines:
                draw_by_page[frame.page_index].text((x1, y_cursor), line, font=font_heading, fill=heading_fill)
                w = text_width(font_heading, line)
                drawn.append(DrawnBox(frame.page_index, [x1, y_cursor, x1 + w, y_cursor + lh], "heading", style.heading_font_size, atom.source_ids, line))
                y_cursor += lh
                mark_usage(frame_idx, y_cursor, len(line))
            y_cursor += gap_after
            mark_usage(frame_idx, y_cursor, 0)
            continue

        lines = wrap_text(atom.text, font_regular, width, first_line_indent="　　")
        lh = line_height(font_regular)
        line_gap = int(style.body_font_size * style.body_line_gap_ratio)
        para_gap = int(style.body_font_size * style.body_para_gap_ratio)

        for line_index, line in enumerate(lines):
            if frame_idx >= len(frames):
                unrendered_chars += len("".join(lines[line_index:]))
                break
            frame = frames[frame_idx]
            x1, y1, x2, y2 = frame.box
            if y_cursor + lh > y2:
                if not move_to_frame(frame_idx + 1):
                    unrendered_chars += len("".join(lines[line_index:]))
                    break
                frame = frames[frame_idx]
                x1, y1, x2, y2 = frame.box
            draw_by_page[frame.page_index].text((x1, y_cursor), line, font=font_regular, fill=fill)
            w = text_width(font_regular, line)
            drawn.append(DrawnBox(frame.page_index, [x1, y_cursor, x1 + w, y_cursor + lh], "body", style.body_font_size, atom.source_ids, line))
            y_cursor += lh + line_gap
            mark_usage(frame_idx, y_cursor, len(line))

        if frame_idx < len(frames):
            y_cursor += para_gap
            mark_usage(frame_idx, y_cursor, 0)

    page_usage: Dict[int, Dict[str, int]] = {}
    for i, frame in enumerate(frames):
        usage = page_usage.setdefault(frame.page_index, {"used": 0, "capacity": 0, "chars": 0})
        usage["used"] += min(frame_used[i], frame_capacity[i])
        usage["capacity"] += frame_capacity[i]
        usage["chars"] += frame_chars[i]

    return drawn, unrendered_chars, page_usage

def intersection_area(a: List[int], b: List[int]) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def overlap_ratio(a: List[int], b: List[int]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1, min(rect_area(a), rect_area(b)))


def drawn_boxes_to_report(
    drawn: List[DrawnBox],
    frames: List[FlowFrame],
    page_usage: Dict[int, Dict[str, int]],
    unrendered_body_chars: int,
    expected_refs: List[str],
    expected_non_body_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rendered_refs = set()
    for box in drawn:
        for source_id in box.source_ids:
            if source_id:
                rendered_refs.add(str(source_id))
    missing_refs = [ref for ref in expected_refs if ref not in rendered_refs]
    expected_non_body_refs = expected_non_body_refs or []
    missing_non_body_refs = [ref for ref in expected_non_body_refs if ref not in rendered_refs]
    truncated_boxes = [
        {"page_index": box.page_index, "role": box.role, "text": box.text[:80], "source_ids": box.source_ids}
        for box in drawn
        if str(box.role).endswith("_truncated")
    ]

    pages: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        pages.setdefault(frame.page_index, {
            "page_index": frame.page_index,
            "body_capacity": 0,
            "body_used": 0,
            "body_chars": 0,
            "drawn_boxes": [],
            "overlaps": [],
            "min_font_size": 999,
            "min_allowed_font_size": 18,
        })
        pages[frame.page_index]["body_capacity"] += max(1, frame.box[3] - frame.box[1])

    for page_index, usage in page_usage.items():
        pages.setdefault(page_index, {
            "page_index": page_index,
            "body_capacity": usage.get("capacity", 1),
            "body_used": 0,
            "body_chars": 0,
            "drawn_boxes": [],
            "overlaps": [],
            "min_font_size": 999,
            "min_allowed_font_size": 18,
        })
        pages[page_index]["body_used"] = max(pages[page_index]["body_used"], usage.get("used", 0))
        pages[page_index]["body_chars"] += usage.get("chars", 0)

    for box in drawn:
        page = pages.setdefault(box.page_index, {
            "page_index": box.page_index,
            "body_capacity": 1,
            "body_used": 0,
            "body_chars": 0,
            "drawn_boxes": [],
            "overlaps": [],
            "min_font_size": 999,
            "min_allowed_font_size": 18,
        })
        page["drawn_boxes"].append({
            "rect": box.rect,
            "role": box.role,
            "font_size": box.font_size,
            "text": box.text[:50],
        })
        if box.role in {"body", "heading"}:
            page["min_font_size"] = min(page["min_font_size"], box.font_size)

    for page in pages.values():
        boxes = page["drawn_boxes"]
        overlaps = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                role_i = boxes[i].get("role")
                role_j = boxes[j].get("role")
                if role_i in {"footer", "kicker"} or role_j in {"footer", "kicker"}:
                    continue
                ratio = overlap_ratio(boxes[i]["rect"], boxes[j]["rect"])
                if ratio > 0.20:
                    overlaps.append({
                        "a": boxes[i]["text"],
                        "b": boxes[j]["text"],
                        "a_role": role_i,
                        "b_role": role_j,
                        "ratio": ratio,
                    })
        page["overlaps"] = overlaps
        page["body_fill_ratio"] = page["body_used"] / max(1, page["body_capacity"])
        if page["min_font_size"] == 999:
            page["min_font_size"] = 0

    return {
        "missing_block_ids": missing_refs,
        "missing_non_body_refs": missing_non_body_refs,
        "truncated_boxes": truncated_boxes,
        "unrendered_body_chars": unrendered_body_chars,
        "pages": list(pages.values()),
    }
