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

logger.info("[Renderer] Initialized with CV2 Rectangular Block Segmentation.")


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
    else:
        x1, y1 = int(coords[0] * img_w / 1000.0), int(coords[1] * img_h / 1000.0)
        x2, y2 = int(coords[2] * img_w / 1000.0), int(coords[3] * img_h / 1000.0)
    x1, x2 = max(0, min(x1, x2)), min(img_w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(img_h, max(y1, y2))
    if (x2 - x1) < 5 or (y2 - y1) < 5: return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# VISUAL DEBUGGING TOOL
# ---------------------------------------------------------------------------

def debug_draw_boxes(image: Image.Image, valid_blocks: list, protected_boxes: list, out_path: str):
    debug_img = isolate(image)
    draw = ImageDraw.Draw(debug_img)
    try:
        debug_font = ImageFont.truetype(FONT_PATH, 20)
    except Exception:
        debug_font = ImageFont.load_default()

    # Draw GPT Text Blocks in Red
    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 25), vb["target_text"][:5] + "...", fill="red", font=debug_font)

    # Draw OpenCV Protected Image Regions in Thick Blue
    for pb in protected_boxes:
        px1, py1, px2, py2 = pb
        draw.rectangle([px1, py1, px2, py2], outline="blue", width=8)
        draw.text((px1 + 10, py1 + 10), "PROTECTED PHOTO REGION", fill="blue", font=debug_font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    debug_img.save(out_path)


# ---------------------------------------------------------------------------
# CV2 RECTANGULAR SEGMENTATION & CANVAS RESET
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


def cv2_rectangular_canvas_reset(image: Image.Image, img_path: str):
    """
    Slices the image into regular rectangles based on OpenCV contour detection.
    Protects large rectangular areas (photos) and wipes everything else.
    """
    arr = np.array(image)
    img_h, img_w = arr.shape[:2]
    temp_dir = "./data/temp"
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Global Base Color
    edges = np.concatenate([
        arr[10:20, 10:-10].reshape(-1, 3), arr[-20:-10, 10:-10].reshape(-1, 3),
        arr[10:-10, 10:20].reshape(-1, 3), arr[10:-10, -20:-10].reshape(-1, 3)
    ])
    global_bg = np.median(edges, axis=0).astype(int)

    # 2. Difference Mask
    diff = np.linalg.norm(arr.astype(float) - global_bg.astype(float), axis=-1)
    raw_mask = (diff > 45).astype(np.uint8) * 255

    # 3. CV2 Morphology (Erode thin text)
    kernel = np.ones((11, 11), np.uint8)
    eroded = cv2.erode(raw_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    # 4. Find Contours and Bounding Rectangles
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    protected_boxes = []
    min_area_threshold = img_w * img_h * 0.015  # Regions must be at least 1.5% of the page area

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area_threshold:
            # Expand the protected box slightly to ensure no image edges are cropped
            px1 = max(0, x - 10)
            py1 = max(0, y - 10)
            px2 = min(img_w, x + w + 10)
            py2 = min(img_h, y + h + 10)
            protected_boxes.append((px1, py1, px2, py2))

    # 5. Build Clean Canvas
    clean_arr = np.full_like(arr, global_bg, dtype=np.uint8)

    # Paste exactly the rectangular protected regions back
    for (px1, py1, px2, py2) in protected_boxes:
        clean_arr[py1:py2, px1:px2] = arr[py1:py2, px1:px2]

    clean_img = Image.fromarray(clean_arr)

    # Save debug canvas to see the perfect rectangular cuts
    clean_img.save(os.path.join(temp_dir, f"debug_clean_blocks_{os.path.basename(img_path)}"))

    return clean_img, global_bg, protected_boxes


# ---------------------------------------------------------------------------
# FONT SIZE FITTING
# ---------------------------------------------------------------------------

def get_optimal_font_size(font_data: bytes, text: str, box_w: int, box_h: int, max_s: int = 100, min_s: int = 8):
    max_s = max(max_s, min_s)
    best_size, best_lines, best_height = min_s, [text], 0
    low, high = min_s, max_s
    line_spacing = 4

    while low <= high:
        mid = (low + high) // 2
        try:
            font = get_font(font_data, mid)
        except Exception:
            return ImageFont.load_default(), [text], 20

        lines, cur_line, cur_h = [], "", 0
        for char in text:
            if font.getbbox(cur_line + char)[2] <= box_w:
                cur_line += char
            else:
                lines.append(cur_line)
                cur_h += (font.getbbox(cur_line)[3] - font.getbbox(cur_line)[1] + line_spacing)
                cur_line = char
        if cur_line:
            lines.append(cur_line)
            cur_h += (font.getbbox(cur_line)[3] - font.getbbox(cur_line)[1] + line_spacing)

        if cur_h <= box_h:
            best_size, best_lines, best_height = mid, lines, cur_h
            low = mid + 1
        else:
            high = mid - 1

    try:
        return get_font(font_data, best_size), best_lines, best_height
    except Exception:
        return ImageFont.load_default(), best_lines, best_height


# ---------------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------------

def renderer_node(state: TranslationState) -> dict:
    logger.info("[Renderer] Starting render process (CV2 Block Segmentation)...")

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

    # 1. CV2 Cut and Reset
    inpainted_image, global_bg, protected_boxes = cv2_rectangular_canvas_reset(image, img_path)

    # 2. Visual Calibration (Now includes the blue protected regions!)
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

        # Calculate the center point of the text block
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Check if the text center falls INSIDE any of the OpenCV protected rectangles
        is_on_photo = False
        for (px1, py1, px2, py2) in protected_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                is_on_photo = True
                break

        pad_w, pad_h = int(bw * 0.05), int(bh * 0.05)
        safe_w, safe_h = bw - 2 * pad_w, bh - 2 * pad_h

        if not is_on_photo:
            # Text is on the background canvas. It's already completely clean.
            bg_for_text = tuple(global_bg)
        else:
            # Text is on the protected photo region. We must wipe it locally.
            local_bg = get_local_bg(orig_arr, x1, y1, x2, y2, img_w, img_h)
            bg_for_text = tuple(local_bg)
            wipe_pad_w, wipe_pad_h = int(bw * 0.05) + 5, int(bh * 0.05) + 5
            draw.rectangle([x1 - wipe_pad_w, y1 - wipe_pad_h, x2 + wipe_pad_w, y2 + wipe_pad_h], fill=bg_for_text)

        lum = (0.299 * bg_for_text[0] + 0.587 * bg_for_text[1] + 0.114 * bg_for_text[2]) / 255
        text_color = (0, 0, 0) if lum > 0.5 else (255, 255, 255)

        font, lines, text_h = get_optimal_font_size(font_data, vb["target_text"], safe_w, safe_h, max_s=int(bh * 0.8))
        y_cursor = y1 + pad_h + (safe_h - text_h) // 2

        for line in lines:
            line_w = font.getbbox(line)[2]
            x_cursor = x1 + pad_w + (safe_w - line_w) // 2
            draw.text((x_cursor, y_cursor), line, font=font, fill=text_color)
            y_cursor += (font.getbbox(line)[3] - font.getbbox(line)[1] + 4)

    out_path = os.path.join(OUTPUT_DIR, f"final_{os.path.basename(img_path)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    inpainted_image.save(out_path)

    logger.info(f"[Renderer] Process complete. Saved to: {out_path}")
    del orig_arr, draw, inpainted_image, valid_blocks
    gc.collect()

    return {"output_image_path": out_path}