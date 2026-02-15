"""
Vietnamese text normalization for ASR evaluation.

Handles:
- Lowercasing
- Punctuation removal / standardization
- Unicode normalization (NFC for Vietnamese diacritics)
- Whitespace normalization
- Common Vietnamese-specific cleaning
"""

import re
import unicodedata


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for ASR evaluation.

    Args:
        text: Raw transcription text.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Unicode NFC normalization (important for Vietnamese diacritics)
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation except Vietnamese-specific characters
    text = _remove_punctuation(text)

    # Normalize common abbreviations and numbers
    text = _normalize_abbreviations(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _remove_punctuation(text: str) -> str:
    """Remove punctuation while preserving Vietnamese characters."""
    # Keep: letters, digits, spaces, Vietnamese diacritics
    # Vietnamese chars are already in Unicode letter categories
    cleaned = []
    for char in text:
        if unicodedata.category(char).startswith(("L", "N")):
            cleaned.append(char)
        elif char in (" ", "\t"):
            cleaned.append(" ")
        # Discard everything else (punctuation, symbols)
    return "".join(cleaned)


def _normalize_abbreviations(text: str) -> str:
    """Normalize common Vietnamese abbreviations and number words."""
    replacements = {
        "tp.": "thành phố",
        "tp ": "thành phố ",
        " tp ": " thành phố ",
        "q.": "quận",
        "p.": "phường",
        "tx.": "thị xã",
        "tt.": "thị trấn",
        "&": "và",
        "%": "phần trăm",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# --- Number-to-word mapping for Vietnamese ---
DIGIT_WORDS = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}


def digits_to_words(text: str) -> str:
    """Convert standalone digits to Vietnamese words.

    Only converts isolated single digits to avoid breaking numbers
    that should remain as sequences.
    """
    def replace_digit(match):
        digit = match.group(0)
        return DIGIT_WORDS.get(digit, digit)

    # Only replace standalone single digits
    return re.sub(r"(?<!\d)(\d)(?!\d)", replace_digit, text)
