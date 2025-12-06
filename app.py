import io
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from torchvision import transforms, models


# ================== PATHS & CONSTANTS ==================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "best_model.pt"
CLASSES_FILE = PROJECT_ROOT / "classes.txt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== MODEL & TRANSFORMS ==================

def load_class_names():
    with open(CLASSES_FILE, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


@st.cache_resource
def load_model_and_classes():
    class_names = load_class_names()
    model = build_model(num_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, class_names


# ================== PREDICTION HELPERS ==================

def predict_image_pil(model, class_names, pil_img: Image.Image):
    tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    return pred_class, probs


def predict_video_file(model, class_names, video_bytes: bytes, frame_skip: int = 3):
    """
    Video prediction: process every `frame_skip`th frame (default = 3).
    """
    tmp_path = PROJECT_ROOT / "tmp_upload_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Could not open the uploaded video file.")

    frame_count = 0
    all_probs = []
    total_frames = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            _, probs = predict_image_pil(model, class_names, pil_img)
            all_probs.append(probs)
            processed_frames += 1

        frame_count += 1

    cap.release()
    tmp_path.unlink(missing_ok=True)

    if len(all_probs) == 0:
        raise RuntimeError(
            "No frames were processed from this video. Try another clip."
        )

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    pred_class = class_names[pred_idx]

    extra_info = {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
    }

    return pred_class, mean_probs, extra_info


def build_explanation(pred_class, probs, class_names, input_type="image"):
    """
    Yahan hum confidence threshold lagate hain.
    Agar confidence < 60% ho to hum bolenge ke
    yeh clear accident / vehicle scene nahi lag raha.
    """
    max_prob = float(np.max(probs))
    confidence_percent = round(max_prob * 100, 2)

    # Threshold: 60%
    if confidence_percent < 60.0:
        # Low confidence -> not sure this is car/accident
        return {
            "input_type": input_type,
            "predicted_class": "Uncertain",
            "confidence_percent": confidence_percent,
            "risk_label": "Not Classified",
            "summary": (
                "The uploaded media does not clearly look like a vehicle / road accident scene. "
                "The model is not confident enough to classify it."
            ),
            "recommendation": (
                "Please upload a clear road or vehicle accident photo/video "
                "(e.g., car crash, damaged vehicle, accident scene) for proper analysis."
            ),
            "is_valid": False,
        }

    # High confidence -> normal behaviour
    if pred_class.lower() == "crashed":
        severity = "High"
        summary = "Accident detected. Vehicle appears damaged."
        recommendation = (
            "Prioritize emergency response. Alert nearby authorities / ambulance. "
            "Use this report as supporting evidence for incident logging."
        )
    else:
        severity = "Low"
        summary = "No visible accident detected in the input."
        recommendation = (
            "Scene looks normal. Still, manual verification is recommended "
            "if this is from live CCTV."
        )

    risk_label = f"{severity} Risk"
    details = {
        "input_type": input_type,
        "predicted_class": pred_class,
        "confidence_percent": confidence_percent,
        "risk_label": risk_label,
        "summary": summary,
        "recommendation": recommendation,
        "is_valid": True,
    }
    return details


# ================== PDF REPORT ==================

def generate_pdf_report(details, probs, class_names, extra_info=None):
    """
    Returns PDF bytes in memory (io.BytesIO) ready for download.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Road Accident Analysis Report")

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Generated on: {now_str}")

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Prediction Summary")

    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Input Type      : {details['input_type'].capitalize()}")
    y -= 16
    c.drawString(60, y, f"Predicted Class : {details['predicted_class']}")
    y -= 16
    c.drawString(60, y, f"Confidence      : {details['confidence_percent']} %")
    y -= 16
    c.drawString(60, y, f"Risk Level      : {details['risk_label']}")

    if extra_info is not None and details["input_type"] == "video":
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Video Details")
        y -= 18
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Total Frames      : {extra_info.get('total_frames', 'N/A')}")
        y -= 16
        c.drawString(60, y, f"Frames Processed  : {extra_info.get('processed_frames', 'N/A')}")

    # Probabilities
    y -= 26
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Class Probabilities")

    y -= 18
    c.setFont("Helvetica", 11)
    for cls, p in zip(class_names, probs):
        c.drawString(60, y, f"{cls:10s}: {round(float(p) * 100, 2):5.2f} %")
        y -= 16
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)

    # Explanation
    y -= 10
    if y < 120:
        c.showPage()
        y = height - 60

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Explanation & Recommendations")

    y -= 20
    c.setFont("Helvetica", 11)
    text_obj = c.beginText(60, y)
    text_obj.textLines(
        f"Summary: {details['summary']}\n\n"
        f"Recommended Actions:\n{details['recommendation']}"
    )
    c.drawText(text_obj)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ================== STREAMLIT UI ==================

def main():
    st.set_page_config(
        page_title="Road Accident Detection & Severity",
        page_icon="🚗",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main {
            background-color: #050816;
            color: #f3f4f6;
        }
        .stButton>button {
            background: linear-gradient(90deg,#2563eb,#7c3aed);
            color: white;
            border-radius: 999px;
            padding: 0.5rem 1.5rem;
            border: none;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg,#22c55e,#eab308,#ef4444);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🚗 Road Accident Detection & Severity Classification")
    st.write(
        "Upload an **image** or **video** to check for accident presence and risk level. "
        "The system uses a trained EfficientNet-based CNN model to analyze the scene."
    )

    model, class_names = load_model_and_classes()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        input_mode = st.radio(
            "Select input type:",
            ("Image", "Video"),
            horizontal=True,
        )

        if input_mode == "Image":
            uploaded_file = st.file_uploader(
                "Upload accident / road scene image",
                type=["jpg", "jpeg", "png"],
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload accident / CCTV video clip",
                type=["mp4", "mov", "avi", "mkv"],
            )

        analyze_button = st.button("Analyze")

    with col_right:
        st.subheader("Preview")
        if uploaded_file is not None:
            if input_mode == "Image":
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, use_column_width=True)
            else:
                st.video(uploaded_file)

    st.markdown("---")

    if analyze_button:
        if uploaded_file is None:
            st.warning("Your analysis has been done. Please upload a file to analyze again.")
            return

        try:
            with st.spinner("Analyzing, please wait..."):
                start_time = time.time()

                if input_mode == "Image":
                    img = Image.open(uploaded_file).convert("RGB")
                    pred_class, probs = predict_image_pil(model, class_names, img)
                    extra_info = None
                    details = build_explanation(pred_class, probs, class_names, input_type="image")
                else:
                    video_bytes = uploaded_file.read()
                    pred_class, probs, extra_info = predict_video_file(
                        model, class_names, video_bytes, frame_skip=3
                    )
                    details = build_explanation(pred_class, probs, class_names, input_type="video")

                elapsed = time.time() - start_time
        except Exception:
            st.error("Your analysis has been done. Please try again with a valid image or video.")
            return

        # Agar model sure nahi hai ke yeh accident/vehicle scene hai
        if not details.get("is_valid", True):
            st.warning(
                "Your analysis has been done, but the system is **not confident** "
                "that this is a road accident / vehicle scene.\n\n"
                "Please upload a clear accident / crash image or video for proper analysis."
            )
            # Yahan hum metrics/report nahi dikhate
            return

        # ====== SHOW RESULTS ======
        st.subheader("Prediction Results")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Predicted Class", details["predicted_class"])
            st.metric("Confidence", f"{details['confidence_percent']} %")
            st.metric("Risk Level", details["risk_label"])
            st.caption(f"Inference time: {elapsed:.2f} seconds")

            if extra_info is not None and input_mode == "Video":
                st.write(
                    f"Processed **{extra_info['processed_frames']}** frames "
                    f"out of **{extra_info['total_frames']}** total."
                )

        with col2:
            st.write("### Class Probabilities")
            for cls, p in zip(class_names, probs):
                # convert numpy float32 -> python float
                p_val = float(p)
                pct = p_val * 100.0
                st.write(f"**{cls}**: {pct:.2f} %")

                # progress bar expects python float between 0 and 1
                bar_val = max(0.0, min(p_val, 1.0))
                st.progress(bar_val)

        st.markdown("### Explanation")
        st.write(details["summary"])
        st.info(details["recommendation"])

        # ====== PDF REPORT DOWNLOAD ======
        pdf_buffer = generate_pdf_report(details, probs, class_names, extra_info)
        file_label = f"accident_report_{input_mode.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=file_label,
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
