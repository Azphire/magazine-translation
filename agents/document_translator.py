from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Tuple

from openai import OpenAI

from agents.translator import contains_bad_english
from core.state import TranslationState
from utils.logger import logger

client = OpenAI()


def load_prompt(filename: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _block_key(page_num: int, block_id) -> str:
    return f"p{page_num}_b{block_id}"


def _ordered_blocks(state: TranslationState) -> List[dict]:
    blocks = (state.get("parsed_json") or {}).get("blocks", []) or []
    return sorted(blocks, key=lambda b: (int(b.get("reading_order", b.get("id", 0))), int(b.get("id", 0))))


def _make_payload(page_states: List[TranslationState]) -> Tuple[List[dict], Dict[str, Tuple[int, dict]]]:
    payload: List[dict] = []
    lookup: Dict[str, Tuple[int, dict]] = {}
    for state in page_states:
        page_num = int(state.get("page_num", 0))
        for b in _ordered_blocks(state):
            key = _block_key(page_num, b.get("id"))
            lookup[key] = (page_num, b)
            payload.append({
                "key": key,
                "page_num": page_num,
                "block_id": b.get("id"),
                "style": b.get("style", "body"),
                "column": b.get("column", 0),
                "reading_order": b.get("reading_order", b.get("id", 0)),
                "text": b.get("text", ""),
            })
    return payload, lookup


NON_BODY_STYLES = {"kicker", "title", "subtitle", "author", "quote", "caption", "footer", "other"}
BODY_STYLES = {"body", "body_heading", "section_heading"}


def _style_requires_block_translation(style: str) -> bool:
    """Return whether a block must be translated as page furniture instead of body flow."""
    return style in NON_BODY_STYLES


def _validate_document_result(result: dict, payload: List[dict]) -> List[str]:
    """Validate translation completeness before rendering.

    This is intentionally strict. Non-body magazine furniture such as pull quotes,
    kickers, captions, footers, and author bios must not silently disappear. Body
    blocks must either appear in global_body_flow or have a block translation for
    non-continuous layouts.
    """
    errors: List[str] = []
    block_trans = result.get("block_translations") or {}
    global_flow = result.get("global_body_flow") or []
    flow_refs = {str(ref) for elem in global_flow for ref in elem.get("source_refs", [])}

    payload_by_key = {str(item.get("key")): item for item in payload}

    for item in payload:
        key = str(item.get("key"))
        style = str(item.get("style", "body"))
        source_text = str(item.get("text", "")).strip()
        if not source_text:
            continue

        if _style_requires_block_translation(style):
            translated = str(block_trans.get(key, "")).strip()
            if not translated:
                errors.append(f"missing non-body translation for {key} style={style}")
            elif contains_bad_english(translated):
                errors.append(f"block_translations[{key}] has English residue: {translated[:100]}")
        elif style in BODY_STYLES:
            if key not in flow_refs and not str(block_trans.get(key, "")).strip():
                errors.append(f"body block {key} is not represented in global_body_flow or block_translations")

    for key, text in block_trans.items():
        translated = str(text).strip()
        if translated and contains_bad_english(translated):
            errors.append(f"block_translations[{key}] has English residue: {translated[:100]}")

    for i, elem in enumerate(global_flow):
        text = str(elem.get("text", "")).strip()
        refs = [str(r) for r in elem.get("source_refs", [])]
        if not text:
            errors.append(f"global_body_flow[{i}] is empty")
        if not refs:
            errors.append(f"global_body_flow[{i}] has no source_refs")
        for ref in refs:
            if ref not in payload_by_key:
                errors.append(f"global_body_flow[{i}] references unknown block {ref}")
        if contains_bad_english(text):
            errors.append(f"global_body_flow[{i}] has English residue: {text[:100]}")
        if elem.get("type") not in ("paragraph", "heading"):
            errors.append(f"global_body_flow[{i}] type must be paragraph or heading")

    return errors


def _translate_missing_blocks(missing_items: List[dict], memory: Dict[str, str], images: List[dict]) -> Dict[str, str]:
    """Retry only missing magazine furniture blocks after the document call.

    A small targeted retry is safer than allowing captions, quotes, or footers to
    vanish. It also keeps the main document translation prompt from becoming too
    large for long magazines.
    """
    if not missing_items:
        return {}

    prompt = (
        "Translate the following magazine page-furniture blocks into concise, publication-quality Chinese.\n"
        "Do not omit any block. Preserve page numbers in footers. For author blocks, use two-line friendly Chinese: "
        "Chinese name（English name）\nChinese bio. Return strict JSON mapping each key to Chinese text.\n\n"
        f"Terminology memory:\n{json.dumps(memory, ensure_ascii=False, indent=2)}\n\n"
        f"Blocks:\n{json.dumps(missing_items, ensure_ascii=False, indent=2)}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + images[:3]}],
            temperature=0.1,
            max_tokens=4096,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        if isinstance(data, dict) and "translations" in data and isinstance(data["translations"], dict):
            data = data["translations"]
        return {str(k): str(v).strip() for k, v in data.items() if str(v).strip()}
    except Exception as exc:
        logger.error(f"[Document Translator] Missing-block retry failed: {exc}", exc_info=True)
        return {}

def translate_document(page_states: List[TranslationState], memory: Dict[str, str], max_retries: int = 2) -> Tuple[List[TranslationState], Dict]:
    """
    Translate the whole article after all pages have been parsed.

    v3 change:
    - GPT sees all body blocks across pages and decides adjacent-page continuity.
    - Continuous body text is translated once into global_body_flow, then the renderer pours it
      through page columns with one fixed font size.
    - Non-body blocks still receive block-level translations.
    """
    payload, lookup = _make_payload(page_states)
    prompt_template = load_prompt("document_translator.txt")
    prompt_text = prompt_template.replace("[MEMORY]", json.dumps(memory, ensure_ascii=False, indent=2)).replace(
        "[DOCUMENT_BLOCKS]", json.dumps(payload, ensure_ascii=False, indent=2)
    )

    images = []
    for state in page_states[:6]:  # keep token/image cost bounded for long documents
        img_path = state.get("image_path")
        if img_path and os.path.exists(img_path):
            images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}})

    last_error = None
    result: Dict = {}
    for attempt in range(max_retries + 1):
        full_prompt = prompt_text
        if last_error:
            full_prompt += f"\n\nPrevious validation errors that MUST be fixed:\n{last_error}\n"
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt}] + images}],
                temperature=0.2,
                max_tokens=12000,
            )
            result = json.loads(response.choices[0].message.content or "{}")
            errors = _validate_document_result(result, payload)
            if not errors:
                break
            last_error = " | ".join(errors)
            logger.warning(f"[Document Translator] validation failed attempt {attempt + 1}: {last_error}")
        except Exception as e:
            last_error = str(e)
            logger.error(f"[Document Translator] failed attempt {attempt + 1}: {e}", exc_info=True)
    else:
        raise RuntimeError(f"Document-level translation failed: {last_error}")

    block_trans = result.get("block_translations") or {}

    # Recover missing non-body blocks after the main document call. This protects
    # pull quotes, captions, footers, and kickers from being dropped silently.
    missing_non_body = []
    for item in payload:
        key = str(item.get("key"))
        style = str(item.get("style", "body"))
        if _style_requires_block_translation(style) and str(item.get("text", "")).strip() and not str(block_trans.get(key, "")).strip():
            missing_non_body.append(item)
    if missing_non_body:
        logger.warning(f"[Document Translator] Retrying {len(missing_non_body)} missing non-body blocks.")
        block_trans.update(_translate_missing_blocks(missing_non_body, memory, images))
        result["block_translations"] = block_trans

    global_flow = result.get("global_body_flow") or []
    continuity = result.get("continuity") or []
    is_continuous = any(bool(item.get("is_continuous")) for item in continuity if isinstance(item, dict))
    if len(page_states) > 1 and not continuity:
        # If GPT omitted continuity but returned a global flow from multiple pages, treat as continuous.
        refs = {ref for elem in global_flow for ref in elem.get("source_refs", [])}
        pages_in_flow = {lookup.get(ref, (None,))[0] for ref in refs if ref in lookup}
        is_continuous = len([p for p in pages_in_flow if p is not None]) > 1

    logger.info(f"[Document Translator] continuity={continuity}; use_global_body_flow={is_continuous}")

    if not global_flow:
        # Emergency fallback: preserve document-level ordering but use block translations.
        for item in payload:
            if item["style"] in ("body", "body_heading", "section_heading"):
                key = item["key"]
                text = str(block_trans.get(key, "")).strip()
                if text:
                    global_flow.append({
                        "type": "heading" if item["style"] in ("body_heading", "section_heading") else "paragraph",
                        "source_refs": [key],
                        "text": text,
                    })

    translated_by_page: Dict[int, List[dict]] = {int(s.get("page_num", 0)): [] for s in page_states}
    body_styles = {"body", "body_heading", "section_heading"}

    for key, (page_num, b) in lookup.items():
        style = b.get("style", "body")
        if is_continuous and style in body_styles:
            # Render body from global_flow only; do not duplicate as per-page body blocks.
            continue
        target_text = str(block_trans.get(key, "")).strip()
        if not target_text:
            if style in body_styles:
                # Non-continuous fallback; keep empty out of renderer rather than falling back to English.
                continue
            logger.warning(f"[Document Translator] Missing non-body translation for {key}; leaving block empty.")
        translated_by_page.setdefault(page_num, []).append({
            "id": b.get("id"),
            "global_key": key,
            "source_text": b.get("text", ""),
            "target_text": target_text,
            "source_box": b.get("source_box"),
            "erase_boxes": b.get("erase_boxes", []),
            "target_box": b.get("target_box"),
            "style": style,
            "column": b.get("column", 0),
            "reading_order": b.get("reading_order", b.get("id", 0)),
            "align": b.get("align", "left"),
            "font_role": b.get("font_role", style),
            "color_role": b.get("color_role", "body"),
            "flow_id": b.get("flow_id", "main"),
        })

    for state in page_states:
        page_num = int(state.get("page_num", 0))
        state["translated_blocks"] = translated_by_page.get(page_num, [])
        state["translator_errors"] = None
        state["translator_retry_count"] = 1

    document_translation = {
        "block_translations": block_trans,
        "global_body_flow": global_flow,
        "continuity": continuity,
        "use_global_body_flow": is_continuous,
    }
    return page_states, document_translation
