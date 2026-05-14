import base64
import os
from typing import List


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def union_boxes(boxes: List[List[int]]) -> List[int]:
    boxes = [b for b in boxes if b and len(b) == 4]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]
