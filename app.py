import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import uuid
import base64
from io import BytesIO
import json

# Load model
model = YOLO("fish_training/fish_detector/weights/best.pt")

st.set_page_config(page_title="Fish Species Detector", layout="wide")
st.title("🐟 Fish Species Detection App")
st.markdown("Upload one or more fish images to detect their species using your trained YOLOv8 model.")

# Sliders and Display Controls
st.sidebar.header("🎛 Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
iou_threshold = st.sidebar.slider("Overlap Threshold (IoU):", 0.0, 1.0, 0.5, 0.01)
opacity_threshold = st.sidebar.slider("Opacity Threshold (for display):", 0, 100, 75)
label_mode = st.sidebar.selectbox("Label Display Mode:", ["Draw Confidence", "Draw Class", "Draw All"])

# Show training graphs
st.sidebar.header("📈 Training Graphs")
graph_files = [
    "results.png", "F1_curve.png", "P_curve.png", "R_curve.png", 
    "confusion_matrix_normalized.png", "labels.jpg"
]

for graph in graph_files:
    path = os.path.join("fish_training", "fish_detector", graph)
    if os.path.exists(path):
        st.sidebar.image(path, caption=graph.split(".")[0], use_column_width=True)

# Upload images
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Helper to convert PIL to base64
def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Center image display
def centered_image(img_base64, caption="", width=500):
    st.markdown(
        f"""
        <div style='text-align: center'>
            <img src='data:image/png;base64,{img_base64}' width="{width}">
            <div style='font-weight: bold; margin-top: 5px'>{caption}</div>
        </div>
        """, unsafe_allow_html=True
    )

# Main prediction loop
if uploaded_files:
    st.info(f"Detecting fish in {len(uploaded_files)} image(s)...")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        img_base64 = pil_to_base64(image)
        centered_image(img_base64, caption=f"Original: {uploaded_file.name}")

        temp_filename = f"{uuid.uuid4()}.jpg"
        image.save(temp_filename)

        results = model.predict(
            source=temp_filename,
            conf=conf_threshold,
            iou=iou_threshold,
            save=True,
            save_txt=False
        )

        output_folder = results[0].save_dir
        predicted_path = os.path.join(output_folder, os.path.basename(temp_filename))

        if os.path.exists(predicted_path):
            pred_img = Image.open(predicted_path).resize((500, 500))
            pred_base64 = pil_to_base64(pred_img)
            centered_image(pred_base64, caption="Prediction", width=500)

            st.subheader("📄 Detection JSON:")
            predictions = []
            for r in results:
                for box in r.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls_id = box
                    cls = model.names[int(cls_id)]
                    prediction = {
                        "x": round((x1 + x2) / 2, 1),
                        "y": round((y1 + y2) / 2, 1),
                        "width": round(x2 - x1),
                        "height": round(y2 - y1),
                        "confidence": round(conf, 3),
                        "class": cls,
                        "class_id": int(cls_id)
                    }
                    predictions.append(prediction)

            st.json({"predictions": predictions})
        else:
            st.warning("⚠ No fish detected.")

        os.remove(temp_filename)
        