import re
import pandas as pd

def decode_unicode_escape(text):
    """Decode unicode escape sequences in the text, ignoring malformed ones."""
    if pd.isna(text):
        return ""
    text = str(text)
    if not text:
        return ""
    # Ignore malformed escape sequences (e.g., truncated \\uXXXX)
    decoded = text.encode("utf-8").decode("unicode_escape", errors="ignore")
    return decoded.encode('ascii', 'ignore').decode('ascii')

def clean_text(text):
    """Clean the text by removing unnecessary whitespace and characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    if not text:
        return ""
    text = decode_unicode_escape(text)
    # Remove non-UTF-8 characters
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    
    return text.strip()