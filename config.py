import os
from dotenv import load_dotenv

load_dotenv()

# Global Configuration
MAX_RETRIES = 3
TEMP_DIR = "./data/temp"
OUTPUT_DIR = "./data/output"

# Typography Configuration
FONT_PATH = "./data/simhei.ttf"

# Inpainting Configuration (Simplified)
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")