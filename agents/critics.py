from core.state import TranslationState
from utils.logger import logger


def parser_critic_node(state: TranslationState) -> dict:
    """
    Validates the structure and geometric logic of the parsed JSON.
    """
    logger.info("[Parser Critic] Validating JSON structure and coordinates.")
    parsed_data = state.get("parsed_json")
    retry_count = state.get("parser_retry_count", 0)

    if not parsed_data or "blocks" not in parsed_data or not parsed_data["blocks"]:
        err = "Error: Missing or empty 'blocks' in output."
        logger.error(f"[Parser Critic] {err}")
        # 返回错误并增加重试计数
        return {"parser_errors": err, "parser_retry_count": retry_count + 1}

    for block in parsed_data["blocks"]:
        box = block.get("box", [])
        # 【新增】：融合了坐标负数越界检查
        if len(box) != 4 or any(v < 0 for v in box):
            err = f"Error: Invalid bounding box format or negative values for block {block.get('id', 'Unknown')}: {box}"
            logger.error(f"[Parser Critic] {err}")
            return {"parser_errors": err, "parser_retry_count": retry_count + 1}

    logger.info("[Parser Critic] Validation passed.")
    # 验证通过，清空错误
    return {"parser_errors": None}


def translator_critic_node(state: TranslationState) -> dict:
    """
    Validates translation fidelity, terminology adherence, missing blocks, and layout safety.
    """
    logger.info("[Translator Critic] Checking translation quality, terminology, and length constraints.")
    translated_blocks = state.get("translated_blocks", [])
    memory = state.get("memory_dict", {})
    retry_count = state.get("translator_retry_count", 0)

    if not translated_blocks:
        err = "Critic Error: No translated blocks found."
        logger.error(f"[Translator Critic] {err}")
        return {"translator_errors": err, "translator_retry_count": retry_count + 1}

    errors = []
    for block in translated_blocks:
        t_text = block.get("target_text", "")
        s_text = block.get("source_text", "")
        b_id = block.get("id", "Unknown")

        # 1. Missing Translation Check (保留你原有的逻辑)
        if not t_text or "[Warning" in t_text:
            errors.append(f"Block {b_id} is missing translation.")
            continue

        # 2. Length Inflation Check (保留你原有的长度溢出校验)
        if len(t_text) > len(s_text) * 1.5:
            errors.append(
                f"Block {b_id} text is too long (Src: {len(s_text)} chars, Trgt: {len(t_text)} chars). "
                f"This will cause layout overflow. Please condense."
            )

        # 3. 【新增】术语忠实度校验 (Terminology Faithfulness Check)
        for en_term, zh_term in memory.items():
            # 如果原文中出现了记忆库里的英文术语（忽略大小写）
            if en_term.lower() in s_text.lower():
                # 译文中必须包含对应的中文术语
                if zh_term not in t_text:
                    errors.append(
                        f"Block {b_id} Terminology Error: Mandatory term '{en_term}' -> '{zh_term}' was NOT used."
                    )

    if errors:
        error_msg = " | ".join(errors)
        logger.warning(f"[Translator Critic] Issues detected: {error_msg}")
        return {"translator_errors": error_msg, "translator_retry_count": retry_count + 1}

    logger.info("[Translator Critic] All translations passed quality, terminology, and layout checks.")
    # 验证通过，清空错误
    return {"translator_errors": None}