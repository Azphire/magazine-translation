import os
import fitz  # PyMuPDF
from PIL import Image

from utils.logger import logger
from core.state import TranslationState
from agents.vision_parser import vision_parser_node
from agents.translator import translator_node
from agents.renderer import renderer_node
from agents.memory import MemoryAgent
from agents.critics import parser_critic_node, translator_critic_node

MAX_RETRIES = 2 

def process_pdf(pdf_path: str, output_pdf_path: str):
    logger.info(f"=== Starting PDF Translation Pipeline: {pdf_path} ===")
    
    # 1. Initialize cross-page memory hub
    memory_agent = MemoryAgent()
    
    doc = fitz.open(pdf_path)
    output_images = []
    temp_dir = "./data/temp"
    os.makedirs(temp_dir, exist_ok=True)

    for page_num in range(len(doc)):
        logger.info(f"\n--- Processing Page {page_num + 1}/{len(doc)} ---")
        page = doc.load_page(page_num)
        
        # Render high-res image (zoom 3.0) for precision OCR
        zoom = 3.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
        pix.save(img_path)

        # -------------------------------------------------------------
        # Optional: Double-page split logic can be inserted here
        # if w > h * 1.2:
        #     # Splitting logic...
        # -------------------------------------------------------------

        state = TranslationState(image_path=img_path)
        state["memory_dict"] = memory_agent.get_memory_context()
        state["parser_retry_count"] = 0
        state["translator_retry_count"] = 0

        # ==========================================
        # Loop 1: Vision Parsing & Critic A
        # ==========================================
        while state.get("parser_retry_count", 0) <= MAX_RETRIES:
            logger.info("[Step 1] Vision Parsing...")
            state.update(vision_parser_node(state))
            
            # Critic A validation
            state.update(parser_critic_node(state))
            if not state.get("parser_errors"):
                break # Passed, exit loop
            logger.info(f"[Retry] Vision Parser triggered retry {state.get('parser_retry_count')}/{MAX_RETRIES}")
            
        # ==========================================
        # Loop 2: Translation & Critic B
        # ==========================================
        while state.get("translator_retry_count", 0) <= MAX_RETRIES:
            logger.info("[Step 2] Multimodal Translating...")
            state.update(translator_node(state))
            
            # Critic B validation
            state.update(translator_critic_node(state))
            if not state.get("translator_errors"):
                break # Passed, exit loop
            logger.info(f"[Retry] Translator triggered retry {state.get('translator_retry_count')}/{MAX_RETRIES}")

        # ==========================================
        # Linear Step 3: Rendering & Memory Update
        # ==========================================
        logger.info("[Step 3] Rendering & Inpainting...")
        state.update(renderer_node(state))

        # Extract entities for the next page
        if "translated_blocks" in state and state["translated_blocks"]:
            memory_agent.update_memory(state["translated_blocks"])

        if "output_image_path" in state:
            output_images.append(state["output_image_path"])

    # Stitch the finalized pages back into a PDF
    if output_images:
        logger.info("\n=== Stitching pages back to PDF ===")
        first_image = Image.open(output_images[0]).convert("RGB")
        rest_images = [Image.open(img).convert("RGB") for img in output_images[1:]]
        
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        first_image.save(output_pdf_path, save_all=True, append_images=rest_images)
        logger.info(f"✅ Pipeline Completed! Final PDF saved to: {output_pdf_path}")

if __name__ == "__main__":
    input_pdf = "./data/input/magazine.pdf"
    output_pdf = "./data/output/magazine_zh.pdf"
    process_pdf(input_pdf, output_pdf)