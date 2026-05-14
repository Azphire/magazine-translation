from langgraph.graph import END, StateGraph

from agents.critics import parser_critic_node, translator_critic_node
from agents.renderer import renderer_node
from agents.translator import translator_node
from agents.vision_parser import vision_parser_node
from config import MAX_RETRIES
from core.state import TranslationState


def check_parser_status(state: TranslationState) -> str:
    if state.get("parser_errors"):
        return "fail" if state.get("parser_retry_count", 0) > MAX_RETRIES else "retry"
    return "pass"


def check_translator_status(state: TranslationState) -> str:
    if state.get("translator_errors"):
        return "fail" if state.get("translator_retry_count", 0) > MAX_RETRIES else "retry"
    return "pass"


def build_translation_graph():
    workflow = StateGraph(TranslationState)
    workflow.add_node("vision_parser", vision_parser_node)
    workflow.add_node("parser_critic", parser_critic_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("translator_critic", translator_critic_node)
    workflow.add_node("renderer", renderer_node)

    workflow.set_entry_point("vision_parser")
    workflow.add_edge("vision_parser", "parser_critic")
    workflow.add_conditional_edges("parser_critic", check_parser_status, {
        "retry": "vision_parser",
        "pass": "translator",
        "fail": END,
    })
    workflow.add_edge("translator", "translator_critic")
    workflow.add_conditional_edges("translator_critic", check_translator_status, {
        "retry": "translator",
        "pass": "renderer",
        "fail": END,
    })
    workflow.add_edge("renderer", END)
    return workflow.compile()
