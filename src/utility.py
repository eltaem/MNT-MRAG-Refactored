import re
def clean_text(text: str) -> str:
    """
    Clean generated text by removing extra whitespace.
    
    Args:
        text: Raw text from captioning model
        
    Returns:
        Cleaned text
    """
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text