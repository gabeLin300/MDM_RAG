from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0


def normalize_encoding(text: object) -> str:
    """Normalize text to UTF-8-safe string and standard newlines."""
    if text is None:
        return ""
    value = str(text)
    if not value:
        return ""
    value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def detect_language(text: object) -> str:
    """Detect language code from text with graceful fallback."""
    value = normalize_encoding(text)
    if not value or len(value) < 5:
        return "unknown"
    try:
        return detect(value)
    except Exception:
        return "unknown"