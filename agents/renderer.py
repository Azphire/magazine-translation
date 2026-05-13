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

logger.info("[Renderer] Initialized with Precision Local Inpainting & Semantic Typography.")

# ---------------------------------------------------------------------------
# IMAGE ISOLATION & FONT CACHING
# ---------------------------------------------------------------------------

def isolate(img: Image.Image) -> Image.Image:
    """Returns a safe copy of the image to prevent memory leaks."""
    return img.copy()

_FONT_CACHE = {}

def get_font(font_data: bytes, size: int):
    """Retrieves or caches a font instance in memory."""
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(io.BytesIO(font_data), size)
    return _FONT_CACHE[size]

def validate_and_fix_box(raw_box: list, img_w: int, img_h: int):
    """Sanitizes absolute pixel coordinates returned by OCR/VLM."""
    if len(raw_box) != 4: return None
    x1, y1, x2, y2 = [int(v) for v in raw_box]
    x1, x2 = max(0, min(x1, x2)), min(img_w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(img_h, max(y1, y2))
    if (x2 - x1) < 5 or (y2 - y1) < 5: return None
    return x1, y1, x2, y2

# ---------------------------------------------------------------------------
# LOCALIZED INPAINTING (The 7x7 Halo Killer)
# ---------------------------------------------------------------------------

def smart_local_inpaint(arr: np.ndarray, valid_blocks: list):
    """
    Generates a targeted mask strictly over text pixels using Otsu thresholding
    and aggressive dilation to eradicate anti-aliased font halos.
    """
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    
    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        roi = arr[y1:y2, x1:x2]
        if roi.size == 0: continue

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Otsu's thresholding automatically separates text from solid backgrounds
        _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # AGGRESSIVE DILATION: 7x7 kernel engulfs all shadowy pixel residues around text
        kernel = np.ones((7, 7), np.uint8) 
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
        # Merge local block mask into the global mask
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], text_mask)

    # Telea Inpainting with a 5px radius to smoothly blend surrounding textures
    inpainted_arr = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
    return inpainted_arr


def get_local_bg(arr: np.ndarray, x1, y1, x2, y2, img_w, img_h, pad=3):
    """Samples the median color of the immediate border surrounding the text box."""
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
    top = arr[by1:max(by1+1, y1), bx1:bx2]
    bottom = arr[min(y2, by2-1):by2, bx1:bx2]
    left = arr[by1:by2, bx1:max(bx1+1, x1)]
    right = arr[by1:by2, min(x2, bx2-1):bx2]
    borders = [b.reshape(-1, 3) for b in (top, bottom, left, right) if b.size > 0]
    return np.median(np.vstack(borders), axis=0).astype(int) if borders else arr[y1, x1].astype(int)


def extract_original_text_color(arr: np.ndarray, x1, y1, x2, y2, bg_color):
    """
    Recovers the original typography color (e.g., red titles) by isolating
    pixels that contrast sharply with the background.
    """
    box_arr = arr[y1:y2, x1:x2]
    if box_arr.size == 0: return (0, 0, 0)
    
    diff = np.linalg.norm(box_arr.astype(float) - np.array(bg_color).astype(float), axis=-1)
    text_pixels = box_arr[diff > 35] 

    if len(text_pixels) > 0:
        return tuple(np.median(text_pixels, axis=0).astype(int))
        
    # Fallback to black or white for optimal readability
    lum = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
    return (0, 0, 0) if lum > 0.5 else (255, 255, 255)

# ---------------------------------------------------------------------------
# ADAPTIVE TYPOGRAPHY (Semantic Spacing)
# ---------------------------------------------------------------------------

def get_adaptive_font_layout(font_data: bytes, text: str, box_w: int, box_h: int, max_s: int=100, min_s: int=8):
    """
    Binary-searches for optimal font size and dynamically redistributes 
    surplus vertical space to line-heights, preventing "floating" paragraphs.
    """
    max_s = max(max_s, min_s)
    best_size, best_lines = min_s, [text]
    best_line_spacing = 4
    best_text_h = 0
    low, high = min_s, max_s

    while low <= high:
        mid = (low + high) // 2
        try: font = get_font(font_data, mid)
        except Exception: return ImageFont.load_default(), [text], 0, 4

        lines, cur_line = [], ""
        for char in text:
            if font.getbbox(cur_line + char)[2] <= box_w:
                cur_line += char
            else:
                if cur_line: lines.append(cur_line)
                cur_line = char
        if cur_line: lines.append(cur_line)

        base_spacing = int(mid * 0.3)
        total_h = sum([font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]) + base_spacing * max(0, len(lines) - 1)

        if total_h <= box_h:
            best_size, best_lines, best_text_h, best_line_spacing = mid, lines, total_h, base_spacing
            low = mid + 1
        else:
            high = mid - 1

    try: final_font = get_font(font_data, best_size)
    except Exception: final_font = ImageFont.load_default()

    if len(best_lines) > 1:
        raw_text_h = sum([final_font.getbbox(line)[3] - final_font.getbbox(line)[1] for line in best_lines])
        empty_space = box_h - raw_text_h
        dynamic_spacing = empty_space // (len(best_lines) - 1)
        max_allowed_spacing = int(best_size * 1.5)
        best_line_spacing = min(dynamic_spacing, max_allowed_spacing)
        best_text_h = raw_text_h + best_line_spacing * (len(best_lines) - 1)

    return final_font, best_lines, best_text_h, best_line_spacing


# ---------------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------------

def renderer_node(state: TranslationState) -> dict:
    logger.info("[Renderer] Starting precision render process...")

    blocks = state.get("translated_blocks", [])
    img_path = state.get("image_path")

    if not os.path.exists(img_path): return {}

    with Image.open(img_path) as im:
        raw_img = im.convert("RGB")
    
    arr = np.array(raw_img)
    img_h, img_w = arr.shape[:2]
    valid_blocks = []

    # Sanitize and prepare blocks
    for b in blocks:
        result = validate_and_fix_box(b.get("box", []), img_w, img_h)
        if result: 
            valid_blocks.append({
                "target_text": b.get("target_text", ""), 
                "coords": result, 
                "w": result[2]-result[0], 
                "h": result[3]-result[1],
                "style": b.get("style", "body") # Retrieve semantic style
            })

    # 1. Localized wipe: destroys English text, protects background & graphics
    inpainted_arr = smart_local_inpaint(arr, valid_blocks)
    inpainted_image = Image.fromarray(inpainted_arr)
    del arr
    gc.collect()

    draw = ImageDraw.Draw(inpainted_image)
    
    try:
        with open(FONT_PATH, "rb") as f: font_data_regular = f.read()
    except IOError: 
        raise FileNotFoundError(f"Font not found: {FONT_PATH}")
    
    # Optional: Load a bold font if available. Otherwise fallback to regular.
    font_data_bold = font_data_regular 

    orig_arr = np.array(raw_img)

    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        bw, bh = vb["w"], vb["h"]
        style = vb["style"]

        # 2. Dynamic Color Recovery
        local_bg = get_local_bg(orig_arr, x1, y1, x2, y2, img_w, img_h)
        text_color = extract_original_text_color(orig_arr, x1, y1, x2, y2, local_bg)

        pad_w, pad_h = int(bw * 0.05), int(bh * 0.05)
        safe_w, safe_h = bw - 2 * pad_w, bh - 2 * pad_h

        # 3. Semantic Rendering Policies
        if style in ["title", "subtitle"]:
            current_font_data = font_data_bold
            max_font_scale = 0.95  # Maximize title size
        elif style == "author":
            current_font_data = font_data_bold
            max_font_scale = 0.8
        else:
            current_font_data = font_data_regular
            max_font_scale = 0.75  # Breathable body text

        font, lines, text_h, dynamic_spacing = get_adaptive_font_layout(
            current_font_data, vb["target_text"], safe_w, safe_h, max_s=int(bh * max_font_scale)
        )
        
        y_cursor = y1 + pad_h + (safe_h - text_h) // 2

        for line in lines:
            line_w = font.getbbox(line)[2]
            line_h = font.getbbox(line)[3] - font.getbbox(line)[1]
            
            # 4. Semantic Alignment
            if style == "body":
                x_cursor = x1 + pad_w # Left-aligned body
            else:
                x_cursor = x1 + pad_w + (safe_w - line_w) // 2 # Center-aligned titles/authors
            
            draw.text((x_cursor, y_cursor), line, font=font, fill=text_color)
            y_cursor += (line_h + dynamic_spacing)

    out_path = os.path.join(OUTPUT_DIR, f"final_{os.path.basename(img_path)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    inpainted_image.save(out_path)

    logger.info(f"[Renderer] Process complete. Saved to: {out_path}")
    del orig_arr, draw, inpainted_image, valid_blocks
    gc.collect()

    return {"output_image_path": out_path}