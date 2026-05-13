import os
import io
import gc
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile

from core.state import TranslationState
from utils.logger import logger
from config import FONT_PATH, OUTPUT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTHONMALLOC"] = "malloc"

logger.info("[Renderer] Initialized with Adaptive Typography & Color Sampling.")


# ---------------------------------------------------------------------------
# SAFER IMAGE ISOLATION
# ---------------------------------------------------------------------------

def isolate(img: Image.Image) -> Image.Image:
    return img.copy()


# ---------------------------------------------------------------------------
# FONT CACHE
# ---------------------------------------------------------------------------

_FONT_CACHE = {}


def get_font(font_data: bytes, size: int):
    key = size
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(io.BytesIO(font_data), size)
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# COORDINATE VALIDATION
# ---------------------------------------------------------------------------

def validate_and_fix_box(raw_box: list, img_w: int, img_h: int):
    if len(raw_box) != 4: return None
    coords = [float(v) for v in raw_box]
    if all(0.0 <= v <= 1.0 for v in coords):
        x1, y1 = int(coords[0] * img_w), int(coords[1] * img_h)
        x2, y2 = int(coords[2] * img_w), int(coords[3] * img_h)
    elif any(v > 1000 for v in coords):
        x1, y1 = int(coords[0]), int(coords[1])
        x2, y2 = int(coords[2]), int(coords[3])
    else:
        x1, y1 = int(coords[0] * img_w / 1000.0), int(coords[1] * img_h / 1000.0)
        x2, y2 = int(coords[2] * img_w / 1000.0), int(coords[3] * img_h / 1000.0)

    x1, x2 = max(0, min(x1, x2)), min(img_w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(img_h, max(y1, y2))
    if (x2 - x1) < 5 or (y2 - y1) < 5: return None
    return x1, y1, x2, y2


def debug_draw_boxes(image: Image.Image, valid_blocks: list, protected_boxes: list, out_path: str):
    debug_img = isolate(image)
    draw = ImageDraw.Draw(debug_img)
    try:
        debug_font = ImageFont.truetype(FONT_PATH, 20)
    except Exception:
        debug_font = ImageFont.load_default()

    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 25), vb["target_text"][:5] + "...", fill="red", font=debug_font)

    for pb in protected_boxes:
        px1, py1, px2, py2 = pb
        draw.rectangle([px1, py1, px2, py2], outline="blue", width=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    debug_img.save(out_path)


# ---------------------------------------------------------------------------
# CV2 BACKGROUND RESET & TEXT COLOR SAMPLING
# ---------------------------------------------------------------------------

def get_local_bg(arr: np.ndarray, x1, y1, x2, y2, img_w, img_h, pad=5):
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
    top = arr[by1:max(by1 + 1, y1), bx1:bx2]
    bottom = arr[min(y2, by2 - 1):by2, bx1:bx2]
    left = arr[by1:by2, bx1:max(bx1 + 1, x1)]
    right = arr[by1:by2, min(x2, bx2 - 1):bx2]
    borders = [b.reshape(-1, 3) for b in (top, bottom, left, right) if b.size > 0]
    return np.median(np.vstack(borders), axis=0).astype(int) if borders else arr[y1, x1].astype(int)


def extract_original_text_color(arr: np.ndarray, x1, y1, x2, y2, bg_color):
    """
    智能原色提取：在裁剪原图框后，找出所有异于背景色的像素，计算中位数作为文字主色。
    """
    box_arr = arr[y1:y2, x1:x2]
    if box_arr.size == 0:
        return (0, 0, 0)

    # 计算框内所有像素与背景色的色差
    diff = np.linalg.norm(box_arr.astype(float) - np.array(bg_color).astype(float), axis=-1)

    # 提取显著异于背景的像素（即原本的文字像素）
    text_pixels = box_arr[diff > 30]

    if len(text_pixels) > 0:
        median_color = np.median(text_pixels, axis=0).astype(int)
        return tuple(median_color)

    # 保底方案：如果没抓到明显的异色（例如文字太细），则基于背景亮度返回黑或白
    lum = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
    return (0, 0, 0) if lum > 0.5 else (255, 255, 255)


def cv2_rectangular_canvas_reset(image: Image.Image, img_path: str):
    arr = np.array(image)
    img_h, img_w = arr.shape[:2]
    temp_dir = "./data/temp"
    os.makedirs(temp_dir, exist_ok=True)

    edges = np.concatenate([
        arr[10:20, 10:-10].reshape(-1, 3), arr[-20:-10, 10:-10].reshape(-1, 3),
        arr[10:-10, 10:20].reshape(-1, 3), arr[10:-10, -20:-10].reshape(-1, 3)
    ])
    global_bg = np.median(edges, axis=0).astype(int)

    diff = np.linalg.norm(arr.astype(float) - global_bg.astype(float), axis=-1)
    raw_mask = (diff > 45).astype(np.uint8) * 255

    kernel = np.ones((11, 11), np.uint8)
    eroded = cv2.erode(raw_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    protected_boxes = []
    min_area_threshold = img_w * img_h * 0.015

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area_threshold:
            px1, py1 = max(0, x - 10), max(0, y - 10)
            px2, py2 = min(img_w, x + w + 10), min(img_h, y + h + 10)
            protected_boxes.append((px1, py1, px2, py2))

    clean_arr = np.full_like(arr, global_bg, dtype=np.uint8)
    for (px1, py1, px2, py2) in protected_boxes:
        clean_arr[py1:py2, px1:px2] = arr[py1:py2, px1:px2]

    clean_img = Image.fromarray(clean_arr)
    return clean_img, global_bg, protected_boxes


# ---------------------------------------------------------------------------
# ADAPTIVE TYPOGRAPHY (行距自适应拉伸算法)
# ---------------------------------------------------------------------------

def get_adaptive_font_layout(font_data: bytes, text: str, box_w: int, box_h: int, max_s: int = 100, min_s: int = 8):
    """
    不仅二分查找最佳字号，还能根据中文较短的特性，自适应拉伸段落行距，让版面饱满。
    """
    max_s = max(max_s, min_s)
    best_size, best_lines = min_s, [text]
    best_line_spacing = 4
    best_text_h = 0

    low, high = min_s, max_s

    # 1. 第一阶段：二分查找能塞进去的最大字号
    while low <= high:
        mid = (low + high) // 2
        try:
            font = get_font(font_data, mid)
        except Exception:
            return ImageFont.load_default(), [text], 0, 4

        lines, cur_line = [], ""
        for char in text:
            if font.getbbox(cur_line + char)[2] <= box_w:
                cur_line += char
            else:
                if cur_line: lines.append(cur_line)
                cur_line = char
        if cur_line: lines.append(cur_line)

        # 基础行距：默认使用字号的 30%
        base_spacing = int(mid * 0.3)
        total_h = sum([font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]) + base_spacing * max(0,
                                                                                                              len(lines) - 1)

        if total_h <= box_h:
            best_size = mid
            best_lines = lines
            best_text_h = total_h
            best_line_spacing = base_spacing
            low = mid + 1
        else:
            high = mid - 1

    # 2. 第二阶段：计算多余垂直空间，动态拉伸行距 (Adaptive Spacing)
    try:
        final_font = get_font(font_data, best_size)
    except Exception:
        final_font = ImageFont.load_default()

    if len(best_lines) > 1:
        # 计算纯文字本身所占的高度
        raw_text_h = sum([final_font.getbbox(line)[3] - final_font.getbbox(line)[1] for line in best_lines])
        empty_space = box_h - raw_text_h

        # 将空闲高度平均分给每两行之间的间隙
        dynamic_spacing = empty_space // (len(best_lines) - 1)

        # 设定拉伸上限：行距最多不能超过字号的 1.5 倍，否则会像散装文字
        max_allowed_spacing = int(best_size * 1.5)
        best_line_spacing = min(dynamic_spacing, max_allowed_spacing)

        # 重新计算最终排版后的总高度
        best_text_h = raw_text_h + best_line_spacing * (len(best_lines) - 1)

    return final_font, best_lines, best_text_h, best_line_spacing


# ---------------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------------

def renderer_node(state: TranslationState) -> dict:
    logger.info("[Renderer] Starting render process (Adaptive Typography)...")

    blocks = state.get("translated_blocks", [])
    img_path = state.get("image_path")

    if not os.path.exists(img_path): return {}

    with Image.open(img_path) as im:
        raw_img = im.convert("RGB")
        raw_img.load()

    image = isolate(raw_img)
    orig_arr = np.array(image)
    del raw_img

    img_w, img_h = image.size
    valid_blocks = []

    for b in blocks:
        result = validate_and_fix_box(b["box"], img_w, img_h)
        if result: valid_blocks.append(
            {"target_text": b["target_text"], "coords": result, "w": result[2] - result[0], "h": result[3] - result[1]})

    inpainted_image, global_bg, protected_boxes = cv2_rectangular_canvas_reset(image, img_path)
    debug_draw_boxes(image, valid_blocks, protected_boxes,
                     os.path.join("./data/temp", f"debug_boxes_{os.path.basename(img_path)}"))
    del image
    gc.collect()

    draw = ImageDraw.Draw(inpainted_image)
    try:
        with open(FONT_PATH, "rb") as f:
            font_data = f.read()
    except IOError:
        raise FileNotFoundError(f"Font not found: {FONT_PATH}")

    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        bw, bh = vb["w"], vb["h"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        is_on_photo = False
        for (px1, py1, px2, py2) in protected_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                is_on_photo = True
                break

        pad_w, pad_h = int(bw * 0.05), int(bh * 0.05)
        safe_w, safe_h = bw - 2 * pad_w, bh - 2 * pad_h

        # 确定底层背景色
        if not is_on_photo:
            bg_for_text = tuple(global_bg)
        else:
            local_bg = get_local_bg(orig_arr, x1, y1, x2, y2, img_w, img_h)
            bg_for_text = tuple(local_bg)
            wipe_pad_w, wipe_pad_h = int(bw * 0.05) + 5, int(bh * 0.05) + 5
            draw.rectangle([x1 - wipe_pad_w, y1 - wipe_pad_h, x2 + wipe_pad_w, y2 + wipe_pad_h], fill=bg_for_text)

        # 【核心新增】：基于原图，智能提取原始英文字体颜色
        text_color = extract_original_text_color(orig_arr, x1, y1, x2, y2, bg_for_text)

        # 【核心新增】：调用自适应排版，获取动态行距
        font, lines, text_h, dynamic_spacing = get_adaptive_font_layout(font_data, vb["target_text"], safe_w, safe_h,
                                                                        max_s=int(bh * 0.8))

        # 整体文字块居中对齐（Y轴起始坐标）
        y_cursor = y1 + pad_h + (safe_h - text_h) // 2

        for line in lines:
            line_w = font.getbbox(line)[2]
            line_h = font.getbbox(line)[3] - font.getbbox(line)[1]

            # X轴水平居中对齐
            x_cursor = x1 + pad_w + (safe_w - line_w) // 2

            draw.text((x_cursor, y_cursor), line, font=font, fill=text_color)

            # 游标下移，使用动态计算的行距
            y_cursor += (line_h + dynamic_spacing)

    out_path = os.path.join(OUTPUT_DIR, f"final_{os.path.basename(img_path)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    inpainted_image.save(out_path)

    logger.info(f"[Renderer] Process complete. Saved to: {out_path}")
    del orig_arr, draw, inpainted_image, valid_blocks
    gc.collect()

    return {"output_image_path": out_path}