import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detect_plate import (
    PlateDetection,
    build_roboflow_client,
    detect_plates,
    load_image,
)
from recognize_plate import build_ocr, recognize_plate_text


# =========================
# Config
# =========================
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_API_KEY = "tmCCkqahpWAKunD3QTFh"
DEFAULT_MODEL_ID = "taiwan-license-plate-recognition-research-tlprr/7"


# =========================
# Utils
# =========================
def save_image(image_path: Path, image: np.ndarray) -> None:
    """儲存圖片（支援中文路徑）"""
    image_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = image_path.suffix or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"無法編碼輸出圖片：{image_path}")

    encoded.tofile(str(image_path))


def draw_detection(image: np.ndarray, detection: PlateDetection, plate_text: str) -> None:
    """畫框 + 文字"""
    x1, y1, x2, y2 = detection.box

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)

    label = plate_text if plate_text else "NO_TEXT"
    label = f"{label} {detection.confidence:.2f}"

    text_y = y1 - 8 if y1 - 8 > 20 else y1 + 24

    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )


# =========================
# Single image pipeline
# =========================
def run_plate_recognition(
    image_path: str,
    api_key: str = DEFAULT_API_KEY,
    api_url: str = DEFAULT_API_URL,
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = "output",
    conf_threshold: float = 0.25,
    ocr_lang: str = "en",
) -> List[str]:

    image = load_image(image_path)
    annotated_image = image.copy()

    roboflow_client = build_roboflow_client(api_key=api_key, api_url=api_url)
    ocr = build_ocr(lang=ocr_lang)

    detections = detect_plates(
        image_source=image_path,
        image=image,
        client=roboflow_client,
        model_id=model_id,
        conf_threshold=conf_threshold,
    )

    plate_texts = []
    output_path = Path(output_dir)

    for idx, detection in enumerate(detections, start=1):

        plate_text, processed_roi, _ = recognize_plate_text(detection.roi, ocr)
        plate_texts.append(plate_text)

        draw_detection(annotated_image, detection, plate_text)

        roi_name = f"{Path(image_path).stem}_plate_{idx:02d}_binary.jpg"
        save_image(output_path / roi_name, processed_roi)

    result_name = f"{Path(image_path).stem}_result.jpg"
    save_image(output_path / result_name, annotated_image)

    return plate_texts


# =========================
# Camera mode
# =========================
def run_camera_recognition(
    camera_index: int = 0,
    api_key: str = DEFAULT_API_KEY,
    api_url: str = DEFAULT_API_URL,
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = "output",
    conf_threshold: float = 0.25,
    ocr_lang: str = "en",
    infer_every: int = 5,
) -> None:

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")

    client = build_roboflow_client(api_key, api_url)
    ocr = build_ocr(ocr_lang)

    last_frame = None
    last_texts = []

    frame_id = 0

    print("Camera started (q=quit, s=save)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1

        if last_frame is None or frame_id % infer_every == 0:
            last_frame, last_texts = process_frame(
                frame, client, ocr, model_id, conf_threshold
            )
        else:
            last_frame = frame.copy()

        cv2.putText(
            last_frame,
            f"Plates: {', '.join(last_texts) if last_texts else 'none'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Plate Detection", last_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame, client, ocr, model_id, conf_threshold):
    annotated = frame.copy()

    detections = detect_plates(
        image_source=frame,
        image=frame,
        client=client,
        model_id=model_id,
        conf_threshold=conf_threshold,
    )

    texts = []
    for d in detections:
        text, _, _ = recognize_plate_text(d.roi, ocr)
        texts.append(text)
        draw_detection(annotated, d, text)

    return annotated, texts


# =========================
# Batch mode (NEW)
# =========================
def run_folder_recognition(
    folder_path: str,
    api_key: str,
    api_url: str,
    model_id: str,
    output_dir: str = "output",
    conf_threshold: float = 0.25,
    ocr_lang: str = "en",
):

    folder = Path(folder_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))

    if not images:
        print("No images found")
        return

    for img in images:
        print(f"\nProcessing: {img.name}")

        texts = run_plate_recognition(
            image_path=str(img),
            api_key=api_key,
            api_url=api_url,
            model_id=model_id,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            ocr_lang=ocr_lang,
        )

        print("Result:", texts if texts else "NO PLATE")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--folder", default=None)

    parser.add_argument("--output", default="output")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-lang", default="en")

    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)

    return parser.parse_args()


# =========================
# Main
# =========================
def main():

    args = parse_args()

    # folder mode
    if args.folder:
        run_folder_recognition(
            folder_path=args.folder,
            api_key=args.api_key,
            api_url=args.api_url,
            model_id=args.model_id,
            output_dir=args.output,
            conf_threshold=args.conf,
            ocr_lang=args.ocr_lang,
        )
        return

    # camera mode
    if args.camera is not None:
        run_camera_recognition(
            camera_index=args.camera,
            api_key=args.api_key,
            api_url=args.api_url,
            model_id=args.model_id,
            output_dir=args.output,
            conf_threshold=args.conf,
            ocr_lang=args.ocr_lang,
        )
        return

    # image mode
    if args.image:
        texts = run_plate_recognition(
            image_path=args.image,
            api_key=args.api_key,
            api_url=args.api_url,
            model_id=args.model_id,
            output_dir=args.output,
            conf_threshold=args.conf,
            ocr_lang=args.ocr_lang,
        )

        print("Result:", texts if texts else "NO PLATE")
        return

    raise SystemExit(
        "請使用:\n"
        "--image xxx.jpg\n"
        "--folder dataset/test/images\n"
        "--camera 0"
    )


if __name__ == "__main__":
    main()