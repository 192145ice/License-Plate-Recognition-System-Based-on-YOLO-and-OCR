import re
import os
from typing import Any, List, Tuple

import cv2
import numpy as np

# Windows CPU 環境下，PaddlePaddle 3.x 的 oneDNN/PIR 有時會出現
# ConvertPirAttribute2RuntimeAttribute 錯誤。車牌 ROI 很小，關閉 MKL-DNN
# 對速度影響不大，且可提高相容性。
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR


def preprocess_plate_roi(roi: np.ndarray) -> np.ndarray:
    """將車牌 ROI 轉為灰階並二值化，提供給 PaddleOCR 辨識。"""

    if roi is None or roi.size == 0:
        raise ValueError("車牌 ROI 為空，無法進行 OCR。")

    # 轉成灰階可降低顏色干擾，讓文字與背景更容易分離。
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 輕微模糊可減少雜訊，避免二值化後產生破碎的小點。
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu 會自動尋找合適閾值，適合光線條件不固定的車牌圖片。
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary


def build_ocr(lang: str = "en", use_angle_cls: bool = True) -> PaddleOCR:
    """建立 PaddleOCR 辨識器，並相容 PaddleOCR v2 與 v3 的參數差異。"""

    # PaddleOCR v3 使用 use_textline_orientation。
    # PaddleOCR v2 使用 use_angle_cls。
    # 這裡逐一嘗試，避免不同版本因未知參數而中斷。
    candidates = [
        {
            "lang": lang,
            "ocr_version": "PP-OCRv4",
            "use_angle_cls": use_angle_cls,
            "show_log": False,
        },
        {
            "lang": lang,
            "ocr_version": "PP-OCRv4",
            "use_textline_orientation": use_angle_cls,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        },
        {
            "lang": lang,
            "ocr_version": "PP-OCRv4",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        },
        {"lang": lang, "use_textline_orientation": use_angle_cls},
        {"lang": lang, "use_angle_cls": use_angle_cls},
        {"lang": lang},
    ]

    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as error:
            last_error = error

    raise RuntimeError(f"無法建立 PaddleOCR：{last_error}")


def _parse_paddle_result(result: Any) -> List[Tuple[str, float]]:
    """將 PaddleOCR 不同版本可能回傳的格式整理成 (文字, 信心分數)。"""

    parsed: List[Tuple[str, float]] = []
    if not result:
        return parsed

    # PaddleOCR v3 常見格式可能包含 rec_texts / rec_scores。
    for item in result if isinstance(result, list) else [result]:
        if isinstance(item, dict):
            texts = item.get("rec_texts") or item.get("texts") or []
            scores = item.get("rec_scores") or item.get("scores") or []
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 0.0
                parsed.append((str(text), score))

    if parsed:
        return parsed

    # PaddleOCR v2 常見格式：
    # [[[[x1,y1],...], ("ABC1234", 0.98)], ...]
    lines = result[0] if isinstance(result, list) and result else result
    if not lines:
        return parsed

    for line in lines:
        try:
            text = str(line[1][0])
            score = float(line[1][1])
        except (IndexError, TypeError, ValueError):
            continue
        parsed.append((text, score))

    return parsed


def normalize_plate_text(text: str) -> str:
    """清理 OCR 結果，只保留車牌常見的英文字母與數字。"""

    # 轉成大寫並移除空白、破折號等非英數字元。
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def recognize_plate_text(
    roi: np.ndarray,
    ocr: PaddleOCR,
) -> Tuple[str, np.ndarray, List[Tuple[str, float]]]:
    """對車牌 ROI 做前處理並使用 PaddleOCR 回傳車牌字串。"""

    processed_roi = preprocess_plate_roi(roi)

    # PaddleOCR v3 需要 3 通道影像，因此將二值圖轉回 BGR。
    # 影像仍然是黑白二值化結果，只是通道數符合 PaddleOCR 輸入格式。
    ocr_input = cv2.cvtColor(processed_roi, cv2.COLOR_GRAY2BGR)

    # 新版 PaddleOCR 不需要 cls=True；角度校正已在建立物件時指定。
    raw_result = ocr.ocr(ocr_input)
    text_scores = _parse_paddle_result(raw_result)

    # 若 OCR 偵測到多段文字，依偵測順序串接後再清理。
    combined_text = "".join(text for text, _ in text_scores)
    plate_text = normalize_plate_text(combined_text)

    return plate_text, processed_roi, text_scores
