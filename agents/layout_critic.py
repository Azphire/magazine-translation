from __future__ import annotations

from typing import Any, Dict, List

from utils.logger import logger


def layout_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic layout critic for automatic re-rendering.

    The critic is deliberately actionable. It does not only say that overlap
    exists; it returns a set of layout directives that the renderer can apply on
    the next attempt, such as moving non-body sidebars to safer boxes and using
    tighter body typography when text overflows.
    """
    report = state.get("layout_report", {}) or {}
    retry_count = int(state.get("layout_retry_count", 0))
    errors: List[str] = []
    suggestions: Dict[str, Any] = {
        "body_font_delta": 0,
        "body_line_gap_delta": 0.0,
        "body_para_gap_delta": 0.0,
        "rebalance_pages": False,
        "allow_more_aggressive_spacing": False,
        "move_authors_to_sidebar": False,
        "force_author_multiline": False,
        "avoid_non_body_overlap": False,
        "expand_non_body_boxes": False,
        "prioritize_content_fit": False,
    }

    missing = report.get("missing_block_ids", []) or []
    if missing:
        errors.append(f"Missing rendered body refs: {missing[:12]}")

    missing_non_body = report.get("missing_non_body_refs", []) or []
    if missing_non_body:
        errors.append(f"Missing rendered non-body refs: {missing_non_body[:12]}")
        suggestions["avoid_non_body_overlap"] = True
        suggestions["expand_non_body_boxes"] = True
        suggestions["prioritize_content_fit"] = True

    truncated = report.get("truncated_boxes", []) or []
    if truncated:
        errors.append(f"Truncated non-body text boxes: {len(truncated)}")
        suggestions["expand_non_body_boxes"] = True
        suggestions["prioritize_content_fit"] = True

    unrendered = int(report.get("unrendered_body_chars", 0) or 0)
    if unrendered > 0:
        errors.append(f"Body overflow: {unrendered} characters were not rendered.")
        suggestions["body_font_delta"] = min(suggestions["body_font_delta"], -2)
        suggestions["body_line_gap_delta"] = min(suggestions["body_line_gap_delta"], -0.08)
        suggestions["body_para_gap_delta"] = min(suggestions["body_para_gap_delta"], -0.10)
        suggestions["rebalance_pages"] = True

    page_fills: List[float] = []
    for page in report.get("pages", []) or []:
        page_index = page.get("page_index")
        fill = float(page.get("body_fill_ratio", 0.0) or 0.0)
        body_chars = int(page.get("body_chars", 0) or 0)
        overlaps = page.get("overlaps", []) or []
        min_font = int(page.get("min_font_size", 0) or 0)
        min_allowed = int(page.get("min_allowed_font_size", 18) or 18)

        if body_chars > 50:
            page_fills.append(fill)

        if overlaps:
            errors.append(f"Page {page_index}: {len(overlaps)} overlaps detected.")
            suggestions["avoid_non_body_overlap"] = True
            if any("author" in (str(item.get("a_role")) + str(item.get("b_role"))) for item in overlaps):
                suggestions["move_authors_to_sidebar"] = True
                suggestions["force_author_multiline"] = True
            # Only shrink body if the overlap involves the body flow.
            if any("body" in {item.get("a_role"), item.get("b_role")} or "heading" in {item.get("a_role"), item.get("b_role")} for item in overlaps):
                suggestions["body_font_delta"] = min(suggestions["body_font_delta"], -1)
                suggestions["body_line_gap_delta"] = min(suggestions["body_line_gap_delta"], -0.03)

        # The fill ratio is now based on vertical column usage, not glyph area.
        if body_chars > 80 and fill < 0.46:
            errors.append(f"Page {page_index}: body fill ratio too low ({fill:.2f}).")
            suggestions["body_font_delta"] = max(suggestions["body_font_delta"], 1)
            suggestions["body_line_gap_delta"] = max(suggestions["body_line_gap_delta"], 0.06)
            suggestions["body_para_gap_delta"] = max(suggestions["body_para_gap_delta"], 0.06)
            suggestions["allow_more_aggressive_spacing"] = True
            suggestions["rebalance_pages"] = True

        if 0 < min_font < min_allowed:
            errors.append(f"Page {page_index}: body font too small ({min_font}px < {min_allowed}px).")
            suggestions["body_font_delta"] = max(suggestions["body_font_delta"], 1)

    if len(page_fills) >= 2 and max(page_fills) - min(page_fills) > 0.45:
        errors.append(f"Body distribution is unbalanced: {[round(v, 2) for v in page_fills]}")
        suggestions["rebalance_pages"] = True
        suggestions["body_line_gap_delta"] = max(suggestions["body_line_gap_delta"], 0.05)

    if errors:
        message = " | ".join(errors)
        logger.warning(f"[Layout Critic] {message}")
        return {
            "layout_errors": message,
            "layout_retry_count": retry_count + 1,
            "layout_suggestions": suggestions,
        }

    logger.info("[Layout Critic] Layout passed deterministic checks.")
    return {
        "layout_errors": None,
        "layout_retry_count": retry_count,
        "layout_suggestions": suggestions,
    }
