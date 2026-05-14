import json
import os
from openai import OpenAI

from config import MEMORY_DIR, PROMPT_DIR
from utils.logger import logger

MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")


def load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


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
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def get_memory_context(self):
        return self.memory

    def update_memory(self, translated_blocks):
        logger.info("[Memory Agent] Extracting cross-page terms...")
        pairs = []
        for b in translated_blocks:
            pairs.append(f"EN: {b.get('source_text', '')}\nZH: {b.get('target_text', '')}")
        combined = "\n\n".join(pairs)
        if not combined.strip():
            return

        prompt = load_prompt("memory_extractor.txt").replace("[COMBINED_TEXT]", combined)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            new_terms = json.loads(response.choices[0].message.content or "{}")
            if new_terms:
                self.memory.update(new_terms)
                self._save_memory()
                logger.info(f"[Memory Agent] Added/updated {len(new_terms)} terms.")
        except Exception as e:
            logger.error(f"[Memory Agent] Failed: {e}", exc_info=True)
