import streamlit as st
import os
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import time
from huggingface_hub import hf_hub_download

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="YOLO Video App", layout="wide")

# ==============================
# Load YOLO Model from Hugging Face
# ==============================
@st.cache_resource
def load_model():
    st.sidebar.info("🔄 Downloading YOLO model from Hugging Face...")

    # Automatically download model file from Hugging Face repo
    model_path = hf_hub_download(
        repo_id="tonett/drowningv1",
        filename="v4.pt"
    )

    model = YOLO(model_path)

    if torch.cuda.is_available():
        model.to("cuda")
        st.sidebar.success("✅ Using GPU for inference")
    else:
        st.sidebar.warning("⚠️ GPU not available, using CPU")
    return model


model = load_model()

# ==============================
# Class Colors (Updated)
# ==============================
class_colors = {
    "drowning": (0, 0, 255),   # 🔴 Red
    "swimming": (0, 255, 0),   # 🟩 Green
}

# ==============================
# Sidebar Mode Selector
# ==============================
mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Video", "Live RTSP Feed"]
)

# ==============================
# Detection Threshold Settings
# ==============================
st.sidebar.markdown("### ⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
iou_threshold = st.sidebar.slider("IoU (Overlap) Threshold", 0.0, 1.0, 0.45, 0.01)

# ==============================
# Mode 1: Upload Video
# ==============================
if mode == "Upload Video":
    st.title("📤 Upload Video for YOLO Detection")

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        input_path = "temp_input.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        save_dir = "saved_results"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(save_dir, f"processed_{timestamp}.mp4")

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        st.markdown(f"**Confidence:** {conf_threshold:.2f} | **IoU:** {iou_threshold:.2f}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                device="cuda" if torch.cuda.is_available() else "cpu",
                conf=conf_threshold,
                iou=iou_threshold,
                stream=True
            )

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    color = class_colors.get(class_name, (0, 255, 0))

                    # Draw bounding box without confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        os.remove(input_path)

        st.success(f"✅ Processing finished! Video saved to {output_path}")

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name=os.path.basename(output_path),
                mime="video/mp4"
            )

# ==============================
# Mode 2: Live RTSP Feed
# ==============================
elif mode == "Live RTSP Feed":
    st.title("📹 Real-time YOLO Detection from RTSP Stream")
    stframe = st.empty()
    alert_placeholder = st.empty()

    save_dir = "livefeed_result"
    os.makedirs(save_dir, exist_ok=True)

    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "video_writer" not in st.session_state:
        st.session_state.video_writer = None

    # Drowning detection time tracking
    if "drowning_start_time" not in st.session_state:
        st.session_state.drowning_start_time = None

    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Start Record", use_container_width=True):
        if not st.session_state.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(save_dir, f"live_record_{timestamp}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            st.session_state.video_writer = cv2.VideoWriter(
                output_path, fourcc, 20.0, (640, 480)
            )
            st.session_state.is_recording = True
            st.sidebar.success(f"🔴 Recording started: {output_path}")

    if col2.button("⏹️ Stop Record", use_container_width=True):
        if st.session_state.is_recording:
            st.session_state.is_recording = False
            if st.session_state.video_writer:
                st.session_state.video_writer.release()
                st.session_state.video_writer = None
            st.sidebar.success("✅ Recording stopped and saved!")

    # Your RTSP camera stream
    rtsp_url = "rtsp://yolov8:yolov8Detection@192.168.137.230:554/stream2"
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        st.error("❌ Cannot open RTSP stream. Check your camera or URL.")
    else:
        st.success("✅ Connected to RTSP stream")
        st.markdown(f"**Confidence:** {conf_threshold:.2f} | **IoU:** {iou_threshold:.2f}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ No frame received from RTSP stream")
                break

            results = model.predict(
                source=frame,
                device="cuda" if torch.cuda.is_available() else "cpu",
                conf=conf_threshold,
                iou=iou_threshold,
                stream=True
            )

            drowning_detected = False

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    color = class_colors.get(class_name, (0, 255, 0))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if class_name.lower() == "drowning":
                        drowning_detected = True

            # Timer logic for drowning (must persist 5 seconds)
            current_time = time.time()
            if drowning_detected:
                if st.session_state.drowning_start_time is None:
                    st.session_state.drowning_start_time = current_time
                elif current_time - st.session_state.drowning_start_time >= 5:
                    alert_placeholder.error("🚨 Someone is DROWNING for 5 seconds! 🚨")
                    st.markdown(
                        """
                        <audio autoplay>
                            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
                        </audio>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.session_state.drowning_start_time = None
                alert_placeholder.empty()

            frame_resized = cv2.resize(frame, (640, 480))
            if st.session_state.is_recording and st.session_state.video_writer:
                st.session_state.video_writer.write(frame_resized)

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

    cap.release()
