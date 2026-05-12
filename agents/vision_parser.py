import os
from core.state import TranslationState
from utils.llm_client import get_gpt4o_client
from utils.image_utils import encode_image_to_base64
from utils.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List


def load_prompt(filename: str) -> str:
    """
    Loads prompt content from the 'prompts' directory.
    Assumes the 'prompts' directory is at the project root.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


# 1. Define the desired JSON structure using Pydantic
class TextBlock(BaseModel):
    id: int = Field(description="Unique ID for the text block based on reading order")
    box: List[int] = Field(
        description="Bounding box [x_min, y_min, x_max, y_max] normalized 0-1000. "
                    "All values must be integers in range [0, 1000]. x_min < x_max, y_min < y_max."
    )
    text: str = Field(description="The exact text content inside the box")


class ParsedLayout(BaseModel):
    blocks: List[TextBlock] = Field(description="List of text blocks ordered by reading flow")


def vision_parser_node(state: TranslationState) -> dict:
    """
    Calls GPT-4o to extract bounding boxes and reading flow from the image.
    Outputs strict JSON matching the ParsedLayout schema.
    """
    img_path = state.get("image_path")
    retry_count = state.get("parser_retry_count", 0)
    errors = state.get("parser_errors")

    logger.info(f"[Vision Parser] Analyzing image layout: {img_path} (Retry: {retry_count})")

    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(img_path)

        # Initialize model and force structured output
        llm = get_gpt4o_client()
        structured_llm = llm.with_structured_output(ParsedLayout)

        # Load base system prompt from text file
        sys_prompt = load_prompt("vision_parser.txt")

        # Append error feedback from Critic if it exists
        if errors:
            sys_prompt += f"\n\nCRITICAL WARNING FROM PREVIOUS ATTEMPT:\n{errors}\nPlease fix this in your next response."
            logger.warning(f"[Vision Parser] Applying error feedback to prompt.")

        # Construct Multimodal Message
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this magazine page and extract the layout."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        # Execute the API Call
        logger.info("[Vision Parser] Waiting for LLM response...")
        response_data = structured_llm.invoke([SystemMessage(content=sys_prompt), message])

        # Convert Pydantic object back to dictionary for state management
        parsed_json = response_data.model_dump()

        logger.info(f"[Vision Parser] Successfully extracted {len(parsed_json.get('blocks', []))} text blocks.")

        return {
            "parsed_json": parsed_json,
            "parser_retry_count": retry_count + 1,
            "parser_errors": None  # Clear errors on successful parse
        }

    except Exception as e:
        logger.error(f"[Vision Parser] LLM API Call Failed: {e}", exc_info=True)
        return {
            "parser_errors": str(e),
            "parser_retry_count": retry_count + 1
        }