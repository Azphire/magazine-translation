import os
import base64
import json
from openai import OpenAI
from utils.logger import logger
from core.state import TranslationState

client = OpenAI()
PROMPT_FILE = "./prompts/vision_translator.txt"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def translator_node(state: TranslationState) -> dict:
    logger.info("[Translator] Starting multi-modal translation...")

    # 【核心修复 1】：正确对齐 vision_parser 输出的数据结构
    parsed_json = state.get("parsed_json", {})
    blocks = parsed_json.get("blocks", [])

    img_path = state.get("image_path")
    memory = state.get("memory_dict", {})

    if not blocks or not img_path:
        logger.warning("[Translator] No text blocks found from parser. Skipping translation.")
        return {"translated_blocks": []}

    base64_image = encode_image(img_path)
    source_texts = {str(i): b["text"] for i, b in enumerate(blocks)}

    # 读取外部 Prompt 模板
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except IOError:
        logger.error(f"[Translator] Prompt file not found: {PROMPT_FILE}")
        return {"translated_blocks": blocks}

    # 注入动态数据
    prompt_text = prompt_template.replace(
        "[MEMORY]", json.dumps(memory, ensure_ascii=False)
    ).replace(
        "[SOURCE_TEXTS]", json.dumps(source_texts, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=4096
        )

        translated_dict = json.loads(response.choices[0].message.content)
        translated_blocks = []

        logger.info("\n=== 🔍 [Translator] Output Review ===")
        for i, b in enumerate(blocks):
            original_text = b["text"]
            translated_text = translated_dict.get(str(i), original_text)

            preview_orig = (original_text[:40] + "...") if len(original_text) > 40 else original_text
            preview_trans = (translated_text[:40] + "...") if len(translated_text) > 40 else translated_text
            logger.info(f"Block [{i}]:")
            logger.info(f"  [EN] {preview_orig.replace(chr(10), ' ')}")
            logger.info(f"  [ZH] {preview_trans.replace(chr(10), ' ')}")

            translated_blocks.append({
                "source_text": original_text,
                "target_text": translated_text,
                "box": b["box"],
                # 【核心修复 2】：使用 .get() 防御 KeyError，因为 parser 模型中未定义 type
                "type": b.get("type", "text")
            })

        logger.info("====================================\n")
        logger.info(f"[Translator] Successfully translated {len(translated_blocks)} blocks.")

        return {"translated_blocks": translated_blocks}

    except Exception as e:
        logger.error(f"[Translator] API Call Failed: {e}")
        return {"translated_blocks": []}