import os
import io
import gc
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFile

from core.state import TranslationState
from utils.logger import logger
from config import STABILITY_API_KEY, FONT_PATH, OUTPUT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTHONMALLOC"] = "malloc"

logger.info("[Renderer] Initialized.")


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
        _FONT_CACHE[key] = ImageFont.truetype(
            io.BytesIO(font_data),
            size
        )
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# COORDINATE VALIDATION
# ---------------------------------------------------------------------------

def validate_and_fix_box(raw_box: list, img_w: int, img_h: int):
    if len(raw_box) != 4:
        return None

    coords = [float(v) for v in raw_box]

    if all(0.0 <= v <= 1.0 for v in coords):
        x1, y1 = int(coords[0] * img_w), int(coords[1] * img_h)
        x2, y2 = int(coords[2] * img_w), int(coords[3] * img_h)
    else:
        x1, y1 = int(coords[0] * img_w / 1000.0), int(coords[1] * img_h / 1000.0)
        x2, y2 = int(coords[2] * img_w / 1000.0), int(coords[3] * img_h / 1000.0)

    x1, x2 = max(0, min(x1, x2)), min(img_w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(img_h, max(y1, y2))

    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# GLOBAL STABILITY AI ERASING (Upgraded from Inpaint)
# ---------------------------------------------------------------------------

def global_inpaint_with_stability(image: Image.Image, valid_blocks: list) -> Image.Image:
    if not STABILITY_API_KEY:
        logger.error("[Renderer] STABILITY_API_KEY missing — skipping inpainting.")
        return isolate(image)

    img_w, img_h = image.size

    # 【核心修复 1】：Stability API 专属遮罩规则
    # 255 (白色) = 保护/保留区域
    # 0 (黑色) = 擦除/消除区域
    mask = Image.new("L", (img_w, img_h), 0)  # 默认全白，保留原背景
    m_draw = ImageDraw.Draw(mask)

    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        m_pad = max(3, int(vb["h"] * 0.08))
        m_draw.rectangle(
            [x1 - m_pad, y1 - m_pad, x2 + m_pad, y2 + m_pad],
            fill=255  # 画黑框，告诉 API "抹掉这里的内容"
        )

    max_dim = 2048

    if max(img_w, img_h) > max_dim:
        scale = max_dim / max(img_w, img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        logger.info(f"[Renderer] Downscaling to {new_w}x{new_h} for API.")

        api_image = isolate(image.resize((new_w, new_h), Image.BICUBIC))
        api_mask = isolate(mask.resize((new_w, new_h), Image.NEAREST))
    else:
        api_image = isolate(image)
        api_mask = isolate(mask)

    def _to_png_bytes(img: Image.Image):
        buf = io.BytesIO()
        isolate(img).save(buf, format="PNG", optimize=False)
        return io.BytesIO(buf.getvalue())

    try:
        # 【核心修复 2】：使用专属的 /erase 接口，彻底舍弃 prompt
        logger.info("[Renderer] Calling Stability AI ERASE API...")

        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/erase",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "image/*"
            },
            files={
                "image": ("image.png", _to_png_bytes(api_image), "image/png"),
                "mask": ("mask.png", _to_png_bytes(api_mask), "image/png"),
            },
            data={"output_format": "png"},  # Erase 接口不需要 Prompt
            timeout=60,
        )

        response.raise_for_status()

        clean = Image.open(io.BytesIO(bytes(response.content))).copy()
        clean = clean.convert("RGB")

        # === 新增：Debug 暂存功能 ===
        temp_dir = "./data/temp"
        os.makedirs(temp_dir, exist_ok=True)

        # 存下遮罩图，方便你确认是不是黑底白框
        api_mask.save(os.path.join(temp_dir, "debug_api_mask.png"))
        # 存下未排版的干净底图
        clean.save(os.path.join(temp_dir, "debug_api_clean_bg.png"))
        logger.info(f"[Renderer] Debug images (mask & clean background) saved to {temp_dir}")
        # ==========================

        if clean.size != (img_w, img_h):
            logger.info(f"[Renderer] Upscaling back to {img_w}x{img_h}.")
            clean = isolate(clean.resize((img_w, img_h), Image.BICUBIC))

        logger.info("[Renderer] Erasing succeeded.")

        del response
        del api_image
        del api_mask
        del mask
        del m_draw
        gc.collect()

        return clean

    except Exception as e:
        logger.warning(f"[Renderer] Stability API failed: {e} — returning original image.")
        gc.collect()
        return isolate(image)


# ---------------------------------------------------------------------------
# FONT SIZE FITTING
# ---------------------------------------------------------------------------

def get_optimal_font_size(
        font_data: bytes,
        text: str,
        box_w: int,
        box_h: int,
        max_s: int = 100,
        min_s: int = 8,
):
    max_s = max(max_s, min_s)
    best_size = min_s
    best_lines = [text]
    best_height = 0
    low, high = min_s, max_s
    line_spacing = 4

    while low <= high:
        mid = (low + high) // 2
        try:
            font = get_font(font_data, mid)
        except Exception:
            return ImageFont.load_default(), [text], 20

        lines = []
        cur_line = ""
        cur_h = 0

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
            best_size = mid
            best_lines = lines
            best_height = cur_h
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
    logger.info("[Renderer] Starting render process...")

    blocks = state.get("translated_blocks", [])
    img_path = state.get("image_path")

    if not os.path.exists(img_path):
        logger.error(f"[Renderer] Source image not found: {img_path}")
        return {}

    with Image.open(img_path) as im:
        raw_img = im.convert("RGB")
        raw_img.load()

    image = isolate(raw_img)
    del raw_img

    img_w, img_h = image.size
    valid_blocks = []

    for b in blocks:
        result = validate_and_fix_box(b["box"], img_w, img_h)
        if result is None:
            logger.warning(f"[Renderer] Skipping invalid box: {b.get('box')}")
            continue
        x1, y1, x2, y2 = result
        valid_blocks.append({
            "target_text": b["target_text"],
            "coords": (x1, y1, x2, y2),
            "w": x2 - x1,
            "h": y2 - y1,
        })

    # Erase Original Text
    inpainted_image = global_inpaint_with_stability(image, valid_blocks)
    del image
    gc.collect()

    # Draw New Text
    draw = ImageDraw.Draw(inpainted_image)

    try:
        with open(FONT_PATH, "rb") as f:
            font_data = f.read()
    except IOError:
        raise FileNotFoundError(f"[Renderer] Font not found: {FONT_PATH}")

    for vb in valid_blocks:
        x1, y1, x2, y2 = vb["coords"]
        bw, bh = vb["w"], vb["h"]

        # Background luminance for text color contrast
        cx = min(max(x1 + bw // 2, 0), img_w - 1)
        cy = min(max(y1 + bh // 2, 0), img_h - 1)
        bg_r, bg_g, bg_b = inpainted_image.getpixel((cx, cy))
        lum = (0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b) / 255
        text_color = (0, 0, 0) if lum > 0.5 else (255, 255, 255)

        pad_w = int(bw * 0.05)
        pad_h = int(bh * 0.05)
        safe_w = bw - 2 * pad_w
        safe_h = bh - 2 * pad_h

        font, lines, text_h = get_optimal_font_size(
            font_data, vb["target_text"], safe_w, safe_h, max_s=int(bh * 0.8)
        )

        y_cursor = y1 + pad_h + (safe_h - text_h) // 2

        for line in lines:
            line_w = font.getbbox(line)[2]
            x_cursor = x1 + pad_w + (safe_w - line_w) // 2
            draw.text((x_cursor, y_cursor), line, font=font, fill=text_color)
            y_cursor += (font.getbbox(line)[3] - font.getbbox(line)[1] + 4)

    out_path = os.path.join(OUTPUT_DIR, f"stability_final_{os.path.basename(img_path)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    inpainted_image.save(out_path)

    logger.info(f"[Renderer] Process complete. Saved to: {out_path}")

    del draw
    del inpainted_image
    del valid_blocks
    gc.collect()

    return {"output_image_path": out_path}