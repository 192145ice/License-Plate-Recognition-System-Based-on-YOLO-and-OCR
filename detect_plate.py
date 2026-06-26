from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient


@dataclass
class PlateDetection:
    """儲存單一車牌偵測結果。"""

    # 車牌框座標，格式為 OpenCV 常用的 (x1, y1, x2, y2)。
    box: Tuple[int, int, int, int]
    # Roboflow 回傳的偵測信心分數。
    confidence: float
    # 從原圖裁切出的車牌 ROI。
    roi: np.ndarray


def load_image(image_path: str) -> np.ndarray:
    """讀取單張圖片，並回傳 OpenCV 使用的 BGR 影像。"""

    # 使用 imdecode 可支援含有中文或特殊字元的 Windows 路徑。
    path = Path(image_path)
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{image_path}")

    return image


def build_roboflow_client(
    api_key: str,
    api_url: str = "https://serverless.roboflow.com",
) -> InferenceHTTPClient:
    """建立 Roboflow Hosted Inference 用戶端。"""

    return InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    )


def _clip_box(
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, int],
    padding: int = 4,
) -> Tuple[int, int, int, int]:
    """將邊界框限制在圖片範圍內，並可加入些微 padding。"""

    height, width = image_shape[:2]
    x1, y1, x2, y2 = box

    # 加入 padding 可讓 OCR 看見完整字元邊緣，避免裁切太緊。
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    return x1, y1, x2, y2


def _prediction_to_xyxy(prediction: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """將 Roboflow 的中心點框格式轉成 (x1, y1, x2, y2)。"""

    # Roboflow Object Detection 通常回傳：
    # {"x": center_x, "y": center_y, "width": w, "height": h, "confidence": score}
    center_x = float(prediction["x"])
    center_y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    x1 = int(round(center_x - width / 2))
    y1 = int(round(center_y - height / 2))
    x2 = int(round(center_x + width / 2))
    y2 = int(round(center_y + height / 2))

    return x1, y1, x2, y2


def detect_plates(
    image_source: Union[str, np.ndarray],
    image: np.ndarray,
    client: InferenceHTTPClient,
    model_id: str = "taiwan-license-plate-recognition-research-tlprr/7",
    conf_threshold: float = 0.25,
    padding: int = 4,
) -> List[PlateDetection]:
    """使用 Roboflow 偵測圖片中的車牌，並自動裁切每個車牌 ROI。"""

    # Roboflow Hosted Inference 可接收圖片路徑；在攝影機模式下則傳入 OpenCV frame。
    result = client.infer(image_source, model_id=model_id)
    predictions = result.get("predictions", []) if isinstance(result, dict) else []

    detections: List[PlateDetection] = []
    for prediction in predictions:
        confidence = float(prediction.get("confidence", 0.0))
        if confidence < conf_threshold:
            continue

        raw_box = _prediction_to_xyxy(prediction)
        clipped_box = _clip_box(raw_box, image.shape, padding)
        x1, y1, x2, y2 = clipped_box

        # 若框不合法，略過此結果。
        if x2 <= x1 or y2 <= y1:
            continue

        roi = image[y1:y2, x1:x2].copy()
        detections.append(
            PlateDetection(
                box=clipped_box,
                confidence=confidence,
                roi=roi,
            )
        )

    # 依信心分數由高到低排序，讓主程式優先處理最可靠的車牌。
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections
