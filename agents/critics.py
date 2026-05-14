from typing import List

from agents.translator import contains_bad_english
from core.state import TranslationState
from utils.layout_utils import box_area, overlap_area
from utils.logger import logger


def _coverage_ratio(lines: List[dict], blocks: List[dict]) -> float:
    if not lines:
        return 1.0
    erase_boxes = []
    for b in blocks:
        erase_boxes.extend(b.get("erase_boxes") or [])
    meaningful = 0
    covered = 0
    for line in lines:
        text = str(line.get("text", "")).strip()
        if len(text) < 2:
            continue
        meaningful += 1
        lb = line.get("box")
        if not lb or box_area(lb) == 0:
            continue
        best = max((overlap_area(lb, eb) / max(1, box_area(lb)) for eb in erase_boxes), default=0)
        if best >= 0.35:
            covered += 1
    return covered / max(1, meaningful)


def parser_critic_node(state: TranslationState) -> dict:
    logger.info("[Parser Critic] Checking OCR/PDF coverage and column safety.")
    parsed = state.get("parsed_json") or {}
    blocks = parsed.get("blocks", [])
    retry_count = int(state.get("parser_retry_count", 0))
    lines = state.get("pdf_text_lines") or state.get("raw_ocr") or []

    if not blocks:
        return {"parser_errors": "Missing parsed_json.blocks.", "parser_retry_count": retry_count + 1}

    errors = []
    for b in blocks:
        for field in ("source_box", "erase_boxes", "target_box", "text", "style", "reading_order"):
            if field not in b:
                errors.append(f"Block {b.get('id')} missing field {field}.")
        sb = b.get("source_box")
        tb = b.get("target_box")
        if not sb or len(sb) != 4 or not tb or len(tb) != 4:
            errors.append(f"Block {b.get('id')} has invalid source_box/target_box.")
        if b.get("style") == "body" and sb and len(sb) == 4:
            # A single body block spanning most of the page width usually means columns were incorrectly merged.
            page_w = max((l.get("box", [0, 0, 0, 0])[2] for l in lines), default=0)
            if page_w and (sb[2] - sb[0]) > page_w * 0.42:
                errors.append(f"Body block {b.get('id')} is too wide and likely merged across columns.")

    coverage = _coverage_ratio(lines, blocks)
    if coverage < 0.82:
        errors.append(f"Only {coverage:.0%} of text lines are assigned to erase_boxes; English residue likely.")

    if errors:
        msg = " | ".join(errors)
        logger.warning(f"[Parser Critic] {msg}")
        return {"parser_errors": msg, "parser_retry_count": retry_count + 1}

    return {"parser_errors": None}


def translator_critic_node(state: TranslationState) -> dict:
    logger.info("[Translator Critic] Checking missing translations, English residue, and terminology.")
    translated_blocks = state.get("translated_blocks", [])
    memory = state.get("memory_dict", {})
    retry_count = int(state.get("translator_retry_count", 0))

    if not translated_blocks:
        return {"translator_errors": "No translated blocks.", "translator_retry_count": retry_count + 1}

    errors = []
    for b in translated_blocks:
        bid = b.get("id")
        source = str(b.get("source_text", ""))
        target = str(b.get("target_text", ""))
        if not target:
            errors.append(f"Block {bid} has empty Chinese translation.")
            continue
        if target.strip() == source.strip():
            errors.append(f"Block {bid} equals original English source.")
        if contains_bad_english(target):
            errors.append(f"Block {bid} contains untranslated English residue: {target[:120]}")
        for en, zh in memory.items():
            if en and zh and en.lower() in source.lower() and zh not in target:
                errors.append(f"Block {bid} must use memory term {en}->{zh}.")

    if errors:
        msg = " | ".join(errors)
        logger.warning(f"[Translator Critic] {msg}")
        return {"translator_errors": msg, "translator_retry_count": retry_count + 1}

    return {"translator_errors": None}
