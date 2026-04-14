from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0

def detect_language(text):
    """Detect the language of the given text. If the text is empty or if detection fails, return 'unknown'."""
    if text.strip() == "":
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"