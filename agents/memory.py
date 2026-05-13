import os
import json
from utils.logger import logger
from openai import OpenAI

MEMORY_FILE = "./data/memory_db/memory.json"

def load_prompt(filename: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "prompts", filename)
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
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

    def get_memory_context(self):
        return self.memory

    def update_memory(self, translated_blocks):
        logger.info("[Memory Agent] Extracting new entities from current page...")

        # Collect all EN-ZH pairs from the current page
        translation_pairs = [f"EN: {b['source_text']} \nZH: {b['target_text']}" for b in translated_blocks]
        combined_text = "\n\n".join(translation_pairs)

        # Load external Prompt template
        try:
            prompt_template = load_prompt("memory_extractor.txt")
        except IOError:
            logger.error("[Memory Agent] Prompt file not found.")
            return

        # Inject dynamic data
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