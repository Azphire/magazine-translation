import base64

def encode_image_to_base64(image_path: str) -> str:
    """
    Reads an image file and encodes it to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string