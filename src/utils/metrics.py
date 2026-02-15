"""
Evaluation metrics for Vietnamese ASR: WER, CER, and dialect-wise analysis.
"""

from jiwer import wer as jiwer_wer, cer as jiwer_cer
from collections import defaultdict

from src.utils.text_normalize import normalize_vietnamese


def compute_wer(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> float:
    """Compute Word Error Rate.

    Args:
        predictions: List of predicted transcriptions.
        references: List of reference transcriptions.
        normalize: Whether to normalize text before computing.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = all errors).
    """
    if normalize:
        predictions = [normalize_vietnamese(p) for p in predictions]
        references = [normalize_vietnamese(r) for r in references]

    # Filter empty references
    filtered = [
        (p, r) for p, r in zip(predictions, references) if r.strip()
    ]
    if not filtered:
        return 0.0

    preds, refs = zip(*filtered)
    return jiwer_wer(list(refs), list(preds))


def compute_cer(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> float:
    """Compute Character Error Rate.

    Args:
        predictions: List of predicted transcriptions.
        references: List of reference transcriptions.
        normalize: Whether to normalize text before computing.

    Returns:
        CER as a float.
    """
    if normalize:
        predictions = [normalize_vietnamese(p) for p in predictions]
        references = [normalize_vietnamese(r) for r in references]

    filtered = [
        (p, r) for p, r in zip(predictions, references) if r.strip()
    ]
    if not filtered:
        return 0.0

    preds, refs = zip(*filtered)
    return jiwer_cer(list(refs), list(preds))


# --- Dialect region mapping ---
DIALECT_REGIONS = {
    "Northern": [
        "Hà Nội", "Hải Phòng", "Quảng Ninh", "Bắc Giang", "Bắc Kạn",
        "Bắc Ninh", "Cao Bằng", "Điện Biên", "Hà Giang", "Hà Nam",
        "Hải Dương", "Hòa Bình", "Hưng Yên", "Lai Châu", "Lạng Sơn",
        "Lào Cai", "Nam Định", "Ninh Bình", "Phú Thọ", "Sơn La",
        "Thái Bình", "Thái Nguyên", "Tuyên Quang", "Vĩnh Phúc", "Yên Bái",
    ],
    "Central": [
        "Đà Nẵng", "Huế", "Thừa Thiên Huế", "Quảng Bình", "Quảng Trị",
        "Quảng Nam", "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa",
        "Ninh Thuận", "Bình Thuận", "Kon Tum", "Gia Lai", "Đắk Lắk",
        "Đắk Nông", "Lâm Đồng", "Hà Tĩnh", "Nghệ An", "Thanh Hóa",
    ],
    "Southern": [
        "Hồ Chí Minh", "TP Hồ Chí Minh", "Bình Dương", "Bình Phước",
        "Đồng Nai", "Tây Ninh", "Bà Rịa - Vũng Tàu", "Long An",
        "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long", "Đồng Tháp",
        "An Giang", "Kiên Giang", "Cần Thơ", "Hậu Giang", "Sóc Trăng",
        "Bạc Liêu", "Cà Mau",
    ],
}


def _province_to_region(province: str) -> str:
    """Map a province name to its dialect region."""
    for region, provinces in DIALECT_REGIONS.items():
        if any(p.lower() in province.lower() for p in provinces):
            return region
    return "Unknown"


def dialect_wise_metrics(
    predictions: list[str],
    references: list[str],
    provinces: list[str],
) -> dict:
    """Compute WER and CER grouped by dialect region.

    Args:
        predictions: List of predicted transcriptions.
        references: List of reference transcriptions.
        provinces: List of province names for each sample.

    Returns:
        Dict with region -> {"wer": float, "cer": float, "count": int}.
    """
    grouped = defaultdict(lambda: {"preds": [], "refs": []})

    for pred, ref, prov in zip(predictions, references, provinces):
        region = _province_to_region(prov)
        grouped[region]["preds"].append(pred)
        grouped[region]["refs"].append(ref)

    results = {}
    for region, data in grouped.items():
        results[region] = {
            "wer": compute_wer(data["preds"], data["refs"]),
            "cer": compute_cer(data["preds"], data["refs"]),
            "count": len(data["preds"]),
        }

    return results
