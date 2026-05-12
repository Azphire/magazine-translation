from core.workflow import build_translation_graph
from utils.logger import logger


def main():
    """
    Entry point for the Multi-Agent Magazine Translation Pipeline.
    """
    logger.info("Initializing Multi-Agent Translation Pipeline...")

    # Compile the graph
    app = build_translation_graph()

    # Prepare the initial state
    initial_state = {
        "image_path": "./data/input/sample_magazine_page.jpg",
        "parser_retry_count": 0,
        "translator_retry_count": 0,
        "memory_dict": {"Agent": "智能体"}  # Mock terminology injection
    }

    logger.info("\n--- Starting Execution ---")

    # Execute the graph
    # LangGraph returns a generator, we iterate through it to see the progress
    for output in app.stream(initial_state):
        # 'output' contains the state updates from the node that just finished
        for node_name, state_update in output.items():
            logger.info(f"Node '{node_name}' finished executing.")

    logger.info("--- Execution Complete ---\n")


if __name__ == "__main__":
    main()