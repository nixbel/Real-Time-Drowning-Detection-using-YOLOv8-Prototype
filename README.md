# 🏊 Real-Time Drowning Detection System — YOLOv8 + Streamlit

> A real-time AI-powered drowning detection application using a custom-trained YOLOv8 model, Streamlit, and OpenCV. Designed to assist lifeguards and pool safety personnel by automatically detecting drowning events from live RTSP camera feeds or uploaded video files.

---

## 📽️ Prototype Demo
 
> Watch the recorded prototype demonstration below:
 
<a href="https://drive.google.com/file/d/1biwlppbYeGulbKaMIOXd-xxYIhp2RuOw/view?usp=sharing" target="_blank">▶ Watch Demo on Google Drive</a>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#️-prototype-demo)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Model Setup (Hugging Face)](#-model-setup-hugging-face)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
  - [Mode 1: Upload Video](#mode-1-upload-video)
  - [Mode 2: Live RTSP Feed](#mode-2-live-rtsp-feed)
- [Detection Classes](#-detection-classes)
- [RTSP Configuration (Tapo Camera)](#-rtsp-configuration-tapo-camera)
- [Output Files](#-output-files)
- [Alert System](#-alarm--alert-system)
- [GPU Support](#-gpu-support)
- [Known Limitations](#-known-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🧠 Overview

This project is a **real-time drowning detection system** built using:

- **YOLOv8** — A state-of-the-art object detection model fine-tuned on a custom drowning/swimming dataset hosted on Hugging Face.
- **Streamlit** — Provides the interactive web-based user interface.
- **OpenCV** — Handles video capture, frame processing, and rendering of bounding boxes.
- **PyTorch** — Powers the model inference, with optional GPU (CUDA) acceleration.

The system supports two operation modes:

1. **Upload Video** — Process a pre-recorded video file and receive a fully annotated output.
2. **Live RTSP Feed** — Connect to an IP camera (e.g., Tapo) for real-time detection with live alerting and optional recording.

---

## ✨ Features

- 🔍 **Real-time object detection** using YOLOv8
- 🎥 **Video upload** support (MP4, AVI, MOV, MKV)
- 📡 **Live RTSP stream** integration (tested with Tapo cameras)
- 🚨 **Drowning alert system** with visual warning and audio alarm
- 🔴 **Live recording** with start/stop controls
- 💾 **Automatic video saving** with timestamped filenames
- ⬇️ **Download processed video** directly from the UI
- ⚡ **GPU acceleration** via CUDA (auto-detected)
- 🤗 **Auto model download** from Hugging Face Hub
- 📊 **Progress bar** for video processing

---

## 📁 Project Structure

```
drowning-detection/
│
├── prototype-hface.py        # Main Streamlit application
│
├── saved_results/            # Auto-created: processed uploaded videos
│   └── processed_YYYYMMDD_HHMMSS.mp4
│
├── livefeed_result/          # Auto-created: recorded live feed clips
│   └── live_record_YYYYMMDD_HHMMSS.mp4
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

> `saved_results/` and `livefeed_result/` are automatically created by the app on first run.

---

## 📦 Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| GPU (optional) | — | NVIDIA CUDA-compatible |
| OS | Windows / Linux / macOS | Ubuntu 20.04+ |

### Python Dependencies

```
streamlit
opencv-python
torch
torchvision
ultralytics
huggingface_hub
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit opencv-python torch torchvision ultralytics huggingface_hub
```

### 4. (Optional) Install CUDA-enabled PyTorch for GPU Support

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your CUDA version. Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🤗 Model Setup (Hugging Face)

The model is automatically downloaded from Hugging Face on first launch. No manual setup needed.

**Model Repository:** [`tonett/drowningv1`](https://huggingface.co/tonett/drowningv1)  
**Model File:** `best.pt`  
**Model Type:** YOLOv8 custom-trained

The download is handled by:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="tonett/drowningv1",
    filename="best.pt",
    repo_type="model"
)
```

The model is **cached locally** by Hugging Face Hub after the first download, so subsequent launches do not require re-downloading.

---

## ▶️ Running the App

```bash
streamlit run prototype-hface.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 📖 Usage Guide

### Mode 1: Upload Video

1. From the **sidebar**, select **"Upload Video"**.
2. Click **"Browse Files"** and select a video (`.mp4`, `.avi`, `.mov`, `.mkv`).
3. The app will begin processing the video **frame by frame**.
4. A **live preview** of the annotated frames will appear, along with a **progress bar**.
5. Once processing is complete:
   - The processed video is automatically saved to the `saved_results/` folder.
   - A **Download** button appears so you can save the file locally.

**Output Example:**

```
saved_results/processed_20250601_143022.mp4
```

---

### Mode 2: Live RTSP Feed

1. From the **sidebar**, select **"Live RTSP Feed"**.
2. The app will attempt to connect to the configured RTSP URL (see [RTSP Configuration](#-rtsp-configuration-tapo-camera)).
3. If connected successfully, the **live annotated stream** will appear in the main view.
4. Use the sidebar buttons to manage recording:
   - **▶️ Start Record** — Begins saving the live feed to `livefeed_result/`.
   - **⏹️ Stop Record** — Stops and finalizes the recording.
5. If a **drowning event** is detected:
   - A red **🚨 alert banner** appears on screen.
   - An **audio alarm** plays automatically.

---

## 🎯 Detection Classes

The model detects two classes:

| Class | Bounding Box Color | Description |
|-------|--------------------|-------------|
| `swimming` | 🔵 Blue `(255, 0, 0)` BGR | Person swimming normally |
| `drowning` | 🟠 Orange `(0, 165, 255)` BGR | Person in distress / drowning |

Label format on bounding box: `classname confidence` (e.g., `drowning 0.87`)

---

## 📡 RTSP Configuration (Tapo Camera)

The RTSP stream URL is hardcoded in the app:

```python
rtsp_url = "rtsp://yolov8:yolov8Detection@192.168.137.230:554/stream2"
```

### To change the camera:

Edit the `rtsp_url` variable in `prototype-hface.py`:

```python
rtsp_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream2"
```

### Tapo Camera RTSP Setup:

1. Open the **Tapo app** on your phone.
2. Go to your camera settings → **Advanced Settings**.
3. Enable **RTSP** and set a username/password.
4. Use `stream1` for high-resolution or `stream2` for lower resolution.

> ⚠️ Make sure your device and camera are on the **same network**.

---

## 💾 Output Files

| Folder | Content | Naming Convention |
|--------|---------|-------------------|
| `saved_results/` | Processed uploaded videos | `processed_YYYYMMDD_HHMMSS.mp4` |
| `livefeed_result/` | Recorded live clips | `live_record_YYYYMMDD_HHMMSS.mp4` |

Both folders are automatically created on first use.

---

## 🚨 Alarm & Alert System

When a `drowning` class is detected in any frame:

- A **red error banner** is displayed:  
  > 🚨 Someone is DROWNING! 🚨

- An **audio alarm** is triggered via an embedded `<audio>` HTML tag using a Google sound source:

```html
<audio autoplay>
  <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
</audio>
```

> ⚠️ The audio autoplay may be blocked in some browsers. Ensure your browser allows autoplay for `localhost`.

---

## ⚡ GPU Support

The app auto-detects CUDA availability:

```python
if torch.cuda.is_available():
    model.to("cuda")
```

- If a **CUDA GPU is found**, inference runs on GPU and a ✅ success message is shown in the sidebar.
- If **no GPU is found**, inference falls back to CPU with a ⚠️ warning.

To verify your CUDA setup:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## ⚠️ Known Limitations

- **Audio alarm** may not autoplay in all browsers due to browser autoplay policies.
- **RTSP URL** is hardcoded; must be manually changed in the source code for different cameras.
- **Live feed resolution** is fixed at `640x480` during recording regardless of camera resolution.
- **No multi-camera support** in the current version.
- **Streamlit re-renders** on every interaction, which may interrupt live stream reconnection.
- Model performance depends on lighting conditions, water clarity, and camera angle.

---

## 🚀 Future Improvements

- [ ] Add a UI input field for dynamic RTSP URL configuration
- [ ] Support multiple camera feeds simultaneously
- [ ] Improve audio alert with local audio file fallback
- [ ] Add SMS / email / push notification integration
- [ ] Log detected drowning events with timestamps to a CSV or database
- [ ] Add configurable confidence threshold slider in the sidebar
- [ ] Implement frame buffering to reduce detection latency
- [ ] Add heatmap or detection history visualization
- [ ] Deploy via Docker for easier cross-platform setup

---

## 📄 License

This project is for **project and research purposes**.

---

## 👤 Author

> Jacques Nico Belmonte - AI Developer

---

*For issues or suggestions, feel free to open a GitHub Issue or reach out directly.*
