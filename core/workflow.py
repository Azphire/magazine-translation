from langgraph.graph import StateGraph, END
from core.state import TranslationState
from agents.vision_parser import vision_parser_node
from agents.critics import parser_critic_node, translator_critic_node
from agents.translator import translator_node
from agents.renderer import renderer_node


def check_parser_status(state: TranslationState) -> str:
    """
    Conditional edge routing for the Parser phase.
    """
    if state.get("parser_errors"):
        # Prevent infinite loops
        if state.get("parser_retry_count", 0) >= 3:
            return "fail"
        return "retry"
    return "pass"


def build_translation_graph():
    """
    Assembles the nodes and edges to build the LangGraph workflow.
    """
    # 1. Initialize Graph with State Definition
    workflow = StateGraph(TranslationState)

    # 2. Add Nodes
    workflow.add_node("vision_parser", vision_parser_node)
    workflow.add_node("parser_critic", parser_critic_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("translator_critic", translator_critic_node)
    workflow.add_node("renderer", renderer_node)

    # 3. Define the Flow (Edges)
    workflow.set_entry_point("vision_parser")
    workflow.add_edge("vision_parser", "parser_critic")

    # 4. Add Conditional Routing (Feedback Loop for Parser)
    workflow.add_conditional_edges(
        "parser_critic",
        check_parser_status,
        {
            "retry": "vision_parser",  # Loop back with error feedback
            "pass": "translator",  # Move to next phase
            "fail": END  # Abort if retries exceeded
        }
    )

    # Define remaining straightforward edges
    workflow.add_edge("translator", "translator_critic")
    # For MVP, assuming translator critic always passes
    workflow.add_edge("translator_critic", "renderer")
    workflow.add_edge("renderer", END)

    # 5. Compile the graph
    return workflow.compile()