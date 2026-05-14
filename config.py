import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MEMORY_DIR = os.path.join(DATA_DIR, "memory_db")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
FONT_DIR = os.path.join(DATA_DIR, "fonts")

MAX_RETRIES = 2
PDF_RENDER_ZOOM = 3.0
DEBUG_SAVE_INTERMEDIATE = True

# Global document translation. v3 parses all pages first, then translates body flow once.
USE_DOCUMENT_LEVEL_TRANSLATION = True
DOCUMENT_TRANSLATION_MODEL = "gpt-4o"

# Keep one fixed body font size across pages in the same article.
# If set to None, renderer computes it from page width once and reuses it.
BODY_FONT_SIZE = None
BODY_LINE_GAP_RATIO = 0.55
BODY_FIRST_LINE_INDENT_EM = 2
BODY_PARAGRAPH_GAP_RATIO = 0.65
BODY_HEADING_GAP_RATIO = 0.45

# Font setup. Put CJK fonts here; otherwise fallback to data/simhei.ttf.
FONT_REGULAR = os.path.join(FONT_DIR, "NotoSansCJK-Regular.otf")
FONT_MEDIUM = os.path.join(FONT_DIR, "NotoSansCJK-Medium.otf")
FONT_BOLD = os.path.join(FONT_DIR, "NotoSansCJK-Bold.otf")
FONT_SERIF = os.path.join(FONT_DIR, "NotoSerifCJK-Regular.otf")
FONT_FALLBACK = os.path.join(FONT_DIR, "simhei.ttf")

COLOR_BODY = (32, 32, 32)
COLOR_TITLE_RED = (180, 45, 55)
COLOR_SUBHEAD_RED = (176, 73, 55)
COLOR_ACCENT_GREEN = (151, 170, 54)
COLOR_MUTED = (84, 84, 84)
COLOR_WHITE = (255, 255, 255)

# Text erasing: PDF-native text boxes first, OCR/parser boxes second, CV text detector last.
ERASE_BOX_PAD_RATIO = 0.44
ERASE_MIN_PAD = 4
ERASE_MAX_PAD = 18
ERASE_INPAINT_RADIUS = 5

# CV fallback detects small black/colored text on white paper even if OCR/PDF misses it.
CV_TEXT_DETECT_ENABLE = True
CV_WHITE_CONTEXT_THRESHOLD = 0.28
CV_DARK_GRAY_THRESHOLD = 205
CV_COMPONENT_MIN_AREA = 12
CV_COMPONENT_MAX_HEIGHT_RATIO = 0.090
CV_COMPONENT_MIN_WIDTH = 4
CV_LINE_DILATE_W = 23
CV_LINE_DILATE_H = 5

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
