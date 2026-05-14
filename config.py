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
MAX_LAYOUT_RETRIES = 3
PDF_RENDER_ZOOM = 3.0
DEBUG_SAVE_INTERMEDIATE = True

# The pipeline parses all pages first, translates the article body as one document,
# and renders the body with one global typographic style.
USE_DOCUMENT_LEVEL_TRANSLATION = True
DOCUMENT_TRANSLATION_MODEL = "gpt-4o"

# Body typography search range. The renderer searches within these ratios and
# chooses a page-balanced size/spacing plan for the whole article.
BODY_FONT_SIZE = None
BODY_MIN_SIZE_RATIO = 0.0105
BODY_MAX_SIZE_RATIO = 0.0160
BODY_TARGET_FILL_RATIO = 0.74
BODY_LINE_GAP_CANDIDATES = [0.20, 0.28, 0.36, 0.44, 0.54, 0.66]
BODY_PARAGRAPH_GAP_CANDIDATES = [0.45, 0.65, 0.85, 1.05, 1.25]
BODY_FIRST_LINE_INDENT_EM = 2
BODY_HEADING_SIZE_RATIO = 1.38
BODY_HEADING_GAP_RATIO = 0.75

# Fonts. The user keeps simhei.ttf inside FONT_DIR, so the fallback path points there.
FONT_REGULAR = os.path.join(FONT_DIR, "NotoSansCJK-Regular.otf")
FONT_MEDIUM = os.path.join(FONT_DIR, "NotoSansCJK-Medium.otf")
FONT_BOLD = os.path.join(FONT_DIR, "NotoSansCJK-Bold.otf")
FONT_SERIF = os.path.join(FONT_DIR, "NotoSerifCJK-Regular.otf")
FONT_FALLBACK = os.path.join(FONT_DIR, "simhei.ttf")

COLOR_BODY = (32, 32, 32)
COLOR_TITLE_RED = (180, 45, 55)
COLOR_SUBHEAD_RED = (176, 73, 55)
COLOR_ACCENT_GREEN = (151, 170, 54)
COLOR_QUOTE_MARK = (198, 106, 82)
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
