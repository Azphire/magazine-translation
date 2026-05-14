from __future__ import annotations

from typing import Dict, List, Tuple


def clamp_box(box: List[int], w: int, h: int) -> List[int] | None:
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [x1, y1, x2, y2]


def box_area(box: List[int]) -> int:
    if not box or len(box) != 4:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap_area(a: List[int], b: List[int]) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def union_boxes(boxes: List[List[int]]) -> List[int]:
    boxes = [b for b in boxes if b and len(b) == 4]
    if not boxes:
        return [0, 0, 0, 0]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def guess_body_column(box: List[int], img_w: int) -> int:
    cx = (box[0] + box[2]) / 2
    left = img_w * 0.36
    mid = img_w * 0.66
    if cx < left:
        return 0
    if cx < mid:
        return 1
    return 2


def detect_large_photo_region(pdf_text_lines: List[dict], img_w: int, img_h: int) -> bool:
    """
    Conservative heuristic: continuation pages in the sample have a large image in top half,
    with most text boxes starting below about 40% page height plus a caption/quote on right.
    This does not detect the image directly; it distinguishes title page vs continuation page.
    """
    if not pdf_text_lines:
        return False
    below = [l for l in pdf_text_lines if l.get("box", [0, 0, 0, 0])[1] > img_h * 0.38]
    above_wide = [l for l in pdf_text_lines if l.get("box", [0, 0, 0, 0])[1] < img_h * 0.34 and (l.get("box", [0, 0, 0, 0])[2] - l.get("box", [0, 0, 0, 0])[0]) > img_w * 0.45]
    return bool(below) and not bool(above_wide)


def infer_chinese_template(img_w: int, img_h: int, page_num: int = 0, has_title: bool = False) -> Dict[str, List[List[int]] | List[int]]:
    """
    UNESCO Courier-like template tuned for the sample article.
    Body columns are fixed page regions; renderer flows Chinese body text through them with one body font size.
    """
    def box(a, b, c, d):
        return [int(img_w * a), int(img_h * b), int(img_w * c), int(img_h * d)]

    if page_num == 0 or has_title:
        return {
            "kicker": box(0.075, 0.030, 0.230, 0.065),
            "title": box(0.075, 0.075, 0.705, 0.205),
            "subtitle": box(0.075, 0.245, 0.730, 0.352),
            "authors": box(0.760, 0.030, 0.945, 0.245),
            "body_columns": [
                box(0.090, 0.375, 0.330, 0.905),
                box(0.385, 0.375, 0.625, 0.748),
                box(0.675, 0.375, 0.915, 0.748),
            ],
            "quote": box(0.385, 0.782, 0.920, 0.915),
            "caption": box(0.070, 0.920, 0.920, 0.955),
            "footer": box(0.065, 0.955, 0.940, 0.988),
        }

    return {
        "caption": box(0.045, 0.025, 0.945, 0.095),
        "quote": box(0.700, 0.130, 0.930, 0.365),
        "body_columns": [
            box(0.090, 0.415, 0.330, 0.895),
            box(0.380, 0.415, 0.620, 0.895),
            box(0.670, 0.415, 0.910, 0.895),
        ],
        "footer": box(0.065, 0.955, 0.940, 0.988),
    }


def target_box_for_style(style: str, img_w: int, img_h: int, page_num: int = 0, has_title: bool = False) -> List[int]:
    tpl = infer_chinese_template(img_w, img_h, page_num, has_title)
    if style == "kicker":
        return tpl.get("kicker", [0, 0, 0, 0])  # type: ignore
    if style == "title":
        return tpl.get("title", [0, 0, 0, 0])  # type: ignore
    if style == "subtitle":
        return tpl.get("subtitle", [0, 0, 0, 0])  # type: ignore
    if style == "author":
        return tpl.get("authors", [0, 0, 0, 0])  # type: ignore
    if style == "quote":
        return tpl.get("quote", [0, 0, 0, 0])  # type: ignore
    if style == "caption":
        return tpl.get("caption", [0, 0, 0, 0])  # type: ignore
    if style == "footer":
        return tpl.get("footer", [0, 0, 0, 0])  # type: ignore
    cols = tpl.get("body_columns", [])
    return cols[0] if cols else [0, 0, img_w, img_h]  # type: ignore
