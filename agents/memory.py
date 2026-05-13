import os
import json
from utils.logger import logger
from openai import OpenAI

MEMORY_FILE = "./data/memory_db/memory.json"
PROMPT_FILE = "./prompts/memory_extractor.txt"


class MemoryAgent:
    def __init__(self):
        self.client = OpenAI()
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_memory(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

    def get_memory_context(self):
        return self.memory

    def update_memory(self, translated_blocks):
        logger.info("[Memory Agent] Extracting new entities from current page...")

        # 收集本页的所有中英对照
        translation_pairs = [f"EN: {b['source_text']} \nZH: {b['target_text']}" for b in translated_blocks]
        combined_text = "\n\n".join(translation_pairs)

        # 读取外部 Prompt 模板
        try:
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except IOError:
            logger.error(f"[Memory Agent] Prompt file not found: {PROMPT_FILE}")
            return

        # 注入动态数据
        prompt = prompt_template.replace("[COMBINED_TEXT]", combined_text)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048
            )

            new_entities = json.loads(response.choices[0].message.content)
            if new_entities:
                self.memory.update(new_entities)
                self._save_memory()
                logger.info(f"[Memory Agent] Successfully added {len(new_entities)} new terms to {MEMORY_FILE}")

        except Exception as e:
            logger.error(f"[Memory Agent] Failed to update memory: {e}")