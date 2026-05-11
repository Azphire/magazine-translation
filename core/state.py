from typing import TypedDict, List, Dict, Optional, Any


class TranslationState(TypedDict):
    """
    Represents the state of the magazine translation pipeline.
    This state is passed between all nodes in the LangGraph.
    """
    # 1. Inputs
    image_path: str

    # 2. Vision Parsing Phase
    parsed_json: Optional[Dict[str, Any]]
    parser_errors: Optional[str]
    parser_retry_count: int

    # 3. Memory Phase
    memory_dict: Dict[str, str]

    # 4. Translation Phase
    translated_blocks: List[Dict[str, Any]]
    translator_errors: Optional[str]
    translator_retry_count: int

    # 5. Output
    output_image_path: Optional[str]