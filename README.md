# 台灣車牌辨識系統

基於 Roboflow 物件偵測與 PaddleOCR 的台灣車牌自動辨識系統，支援單張圖片、資料夾批次處理與即時攝影機辨識三種模式。

---

## 功能特色

- **車牌偵測**：透過 Roboflow Hosted Inference 呼叫預訓練模型（`taiwan-license-plate-recognition-research-tlprr/7`），定位圖片中的車牌位置。
- **影像前處理**：對裁切出的車牌 ROI 進行灰階轉換、高斯模糊與 Otsu 二值化，提升 OCR 辨識率。
- **文字辨識**：使用 PaddleOCR（PP-OCRv4）辨識車牌文字，相容 v2 / v3 兩種版本 API。
- **結果輸出**：標注偵測框與車牌號碼後儲存結果圖，並另外輸出每張車牌的二值化 ROI。
- **三種執行模式**：單張圖片、整個資料夾批次、即時攝影機串流。

---

## 專案結構

```
.
├── detect_plate.py       # 車牌偵測模組（Roboflow 推論、ROI 裁切）
├── recognize_plate.py    # 車牌辨識模組（影像前處理、PaddleOCR）
├── main.py               # 主程式入口（CLI 解析、三種執行模式）
├── output/               # 輸出目錄
└── dataset/test/images   # 測試圖片
```

---

## 環境需求

| 套件 | 說明 |
|------|------|
| `opencv-python` | 影像讀取、前處理與標注 |
| `numpy` | 陣列操作 |
| `inference-sdk` | Roboflow Hosted Inference 用戶端 |
| `paddlepaddle` | PaddleOCR 依賴的深度學習框架 |
| `paddleocr` | OCR 引擎（建議 PP-OCRv4） |

安裝方式：

```bash
pip install opencv-python numpy inference-sdk paddlepaddle paddleocr
```

> **注意（Windows CPU）**：PaddleOCR 在部分 Windows 環境下可能出現 MKL-DNN 相容性問題，系統已自動設定 `FLAGS_use_mkldnn=0` 以提高穩定性。

---

## 快速開始

### 單張圖片

```bash
python main.py --image car.jpg
```

### 資料夾批次處理

```bash
python main.py --folder dataset/test/images
```

### 即時攝影機

```bash
python main.py --camera 0
```

---

## 完整 CLI 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--image` | — | 單張圖片路徑 |
| `--folder` | — | 資料夾路徑（批次模式） |
| `--camera` | — | 攝影機索引（即時模式） |
| `--output` | `output` | 結果輸出目錄 |
| `--conf` | `0.25` | 車牌偵測信心閾值 |
| `--ocr-lang` | `en` | PaddleOCR 辨識語言 |
| `--api-key` | *(內建)* | Roboflow API 金鑰 |
| `--api-url` | `https://serverless.roboflow.com` | Roboflow API 端點 |
| `--model-id` | `taiwan-license-plate-recognition-research-tlprr/7` | Roboflow 模型 ID |

---

## 輸出說明

每次辨識完成後，會在 `--output` 目錄下產生以下檔案：

| 檔案命名格式 | 內容 |
|-------------|------|
| `{原檔名}_result.jpg` | 標注偵測框與車牌號碼的完整圖片 |
| `{原檔名}_plate_{N:02d}_binary.jpg` | 第 N 張車牌的二值化 ROI |

---

## 模組說明

### `detect_plate.py`

負責與 Roboflow API 溝通並裁切車牌區域。

- `load_image(path)` — 讀取圖片，支援中文與特殊字元路徑。
- `build_roboflow_client(api_key, api_url)` — 建立 Roboflow 推論用戶端。
- `detect_plates(image_source, image, client, ...)` — 呼叫模型進行偵測，回傳依信心分數排序的 `PlateDetection` 清單，每個物件包含 `box`、`confidence` 與裁切出的 `roi`。

### `recognize_plate.py`

負責影像前處理與 OCR 辨識。

- `preprocess_plate_roi(roi)` — 灰階 → 高斯模糊 → Otsu 二值化。
- `build_ocr(lang, use_angle_cls)` — 建立 PaddleOCR，自動相容 v2 / v3 參數差異。
- `recognize_plate_text(roi, ocr)` — 對 ROI 執行 OCR 並回傳正規化後的車牌字串（僅保留大寫英數字元）。

### `main.py`

整合偵測與辨識流程，提供三種執行模式：

- `run_plate_recognition` — 單張圖片完整流程。
- `run_folder_recognition` — 遍歷資料夾中所有 `.jpg` / `.png` / `.jpeg` 圖片。
- `run_camera_recognition` — 讀取攝影機串流，每隔 N 幀（預設 5 幀）執行一次推論以平衡效能。

---

## 注意事項

- Roboflow API 金鑰已內建於程式碼中，正式部署前請改以環境變數管理，避免洩漏。
- 攝影機模式下按 `q` 鍵退出。
- OCR 後處理會移除所有非英數字元（空格、破折號等），如需支援其他格式請修改 `normalize_plate_text()`。
