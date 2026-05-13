import os
import json
import logging

from core.state import TranslationState
from utils.llm_client import get_gpt4o_client
from utils.image_utils import encode_image_to_base64
from utils.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List

# ---------------------------------------------------------------------------
# OCR ENGINE — RapidOCR (ONNX-based, no C++ runtime, stable on Windows)
# ---------------------------------------------------------------------------
from rapidocr_onnxruntime import RapidOCR

# Suppress verbose OCR logs
logging.getLogger("rapidocr").setLevel(logging.ERROR)

# Initialize global OCR engine
ocr_engine = RapidOCR()


def load_prompt(filename: str) -> str:
    """Loads prompt content from the 'prompts' directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMA
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    id: int = Field(description="Unique ID for the merged text block")
    box: List[int] = Field(
        description="Bounding box [x_min, y_min, x_max, y_max] in ABSOLUTE PIXELS."
    )
    text: str = Field(description="The merged exact text content")
    style: str = Field(description="Semantic style: title, subtitle, author, body, quote, caption")


class ParsedLayout(BaseModel):
    blocks: List[TextBlock] = Field(
        description="List of merged text blocks ordered by reading flow"
    )


# ---------------------------------------------------------------------------
# RAW OCR EXTRACTION
# ---------------------------------------------------------------------------

def extract_raw_ocr(img_path: str) -> list:
    """
    Run RapidOCR on the image and return a flat list of detected text lines
    with axis-aligned bounding rectangles.
    """
    result, _ = ocr_engine(img_path)
    ocr_blocks = []
    
    if not result:
        logger.warning("[Vision Parser] RapidOCR returned no results.")
        return ocr_blocks

    for idx, line in enumerate(result):
        box, text, score = line[0], line[1], line[2]
        x_min = int(min(pt[0] for pt in box))
        y_min = int(min(pt[1] for pt in box))
        x_max = int(max(pt[0] for pt in box))
        y_max = int(max(pt[1] for pt in box))
        ocr_blocks.append({
            "id":   idx,
            "box":  [x_min, y_min, x_max, y_max],
            "text": text,
            "score": round(float(score), 3),
        })

    return ocr_blocks


# ---------------------------------------------------------------------------
# MAIN NODE
# ---------------------------------------------------------------------------

def vision_parser_node(state: TranslationState) -> dict:
    """
    Executes the Hybrid-RAG pipeline: OCR for precision, VLM for semantic merging, 
    drop-cap stitching, and missing-title recovery.
    """
    img_path    = state.get("image_path")
    retry_count = state.get("parser_retry_count", 0)
    errors      = state.get("parser_errors")

    logger.info(f"[Vision Parser] RapidOCR + GPT-4o hybrid analysis: {img_path} (retry={retry_count})")

    try:
        # 1. Pixel-accurate OCR
        raw_ocr_data = extract_raw_ocr(img_path)
        logger.info(f"[Vision Parser] RapidOCR extracted {len(raw_ocr_data)} raw lines.")

        # 2. Encode image for the VLM
        base64_image = encode_image_to_base64(img_path)

        # 3. Build structured LLM call
        llm            = get_gpt4o_client()
        structured_llm = llm.with_structured_output(ParsedLayout)
        sys_prompt     = load_prompt("vision_parser.txt")

        # Inject previous critic feedback if any
        if errors:
            sys_prompt += (
                f"\n\nCRITICAL WARNING FROM PREVIOUS ATTEMPT:\n{errors}\n"
                "Please fix this in your next response."
            )
            logger.warning("[Vision Parser] Injecting error feedback into prompt.")

        ocr_json_str = json.dumps(raw_ocr_data, ensure_ascii=False)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"Here is the raw OCR data extracted from the image:\n{ocr_json_str}\n\n"
                        "Using both the OCR data and the image, merge adjacent lines, fix drop caps, "
                        "rescue missing artistic titles, and return structured styled blocks."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # 4. Call VLM
        logger.info("[Vision Parser] Waiting for VLM to merge and structure paragraphs...")
        response_data = structured_llm.invoke([SystemMessage(content=sys_prompt), message])
        parsed_json   = response_data.model_dump()

        logger.info(f"[Vision Parser] Merged into {len(parsed_json.get('blocks', []))} semantic blocks.")

        return {
            "parsed_json":          parsed_json,
            "parser_retry_count":   retry_count + 1,
            "parser_errors":        None,
        }

    except Exception as e:
        logger.error(f"[Vision Parser] Hybrid analysis failed: {e}", exc_info=True)
        return {
            "parser_errors":      str(e),
            "parser_retry_count": retry_count + 1,
        }