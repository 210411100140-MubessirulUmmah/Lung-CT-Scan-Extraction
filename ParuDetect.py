import streamlit as st
import logging
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile
import json
from ultralytics import YOLO
import supervision as sv

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.WARNING)

# Streamlit page config with custom styling
st.set_page_config(page_title="Lung CT-Scan Extraction", page_icon="🫁", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #111, #333);
        color: white;
    }
    .block-container {
        text-align: center;
        max-width: 700px;
        margin: auto;
    }
    h1, h2 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;  /* Menghilangkan border default */
        transition: background-color 0.3s ease;  /* Animasi saat hover */
    }
    .stButton button:hover {
        background-color: #2ca02c;
    }
    .stButton {
        margin: 10px 0;  /* Jarak antara tombol */
    }
    .download-all-btn {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model("best14.pt")
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def save_detections_to_json(detections, file_name):
    data = [{"box": box.tolist(), "confidence": float(conf), "class_id": int(cid)}
            for box, conf, cid in zip(detections.xyxy, detections.confidence, detections.class_id)]
    json_str = json.dumps(data)
    return json_str

def download_zip(images, filenames, zip_name="detections.zip"):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for img, name in zip(images, filenames):
            img_buffer = BytesIO()
            img.save(img_buffer, format="JPEG")
            zf.writestr(name, img_buffer.getvalue())
    zip_buffer.seek(0)
    st.download_button(label="Download All Detections", data=zip_buffer, file_name=zip_name, key='download_zip')

def process_and_display_image(image, conf):
    results = model.predict(image)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = detections[detections.confidence > conf]
    labels = [f"{results[0].names[cid]} ({conf:.2f})" 
              for cid, conf in zip(detections.class_id, detections.confidence)]

    annotated_img = box_annotator.annotate(image, detections=detections)
    annotated_img = label_annotator.annotate(annotated_img, detections=detections, labels=labels)
    
    img_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    return img_pil, labels, save_detections_to_json(detections, "detections.json")

def main():
    st.title("Lung CT-Scan Extraction")
    st.sidebar.title("Select an option ⤵️")
    choice = st.sidebar.radio("", ("Capture Image And Predict", "Upload Multiple Images 📂"))
    conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)

    if choice == "Upload Multiple Images 📂":
        uploaded_files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            st.markdown("### Detection Results")
            col1, col2, col3 = st.columns(3)  # Display images in 3 columns
            detected_images, filenames = [], []

            for idx, uploaded_file in enumerate(uploaded_files):
                img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                img_pil, labels, json_data = process_and_display_image(img, conf)

                detected_images.append(img_pil)
                filenames.append(f"{uploaded_file.name}_detected.jpg")

                # Display in columns
                with [col1, col2, col3][idx % 3]:
                    st.image(img_pil, caption=uploaded_file.name, use_column_width=True)
                    st.download_button(label="Download", data=img_pil.tobytes(),
                                       file_name=f"{uploaded_file.name}_detected.jpg", mime="image/jpeg")

            # Button to download all images as a ZIP
            download_zip(detected_images, filenames)

    elif choice == "Capture Image And Predict":
        img_buffer = st.camera_input("Capture a Chest X-Ray")
        if img_buffer:
            img = cv2.imdecode(np.frombuffer(img_buffer.read(), np.uint8), cv2.IMREAD_COLOR)
            img_pil, labels, json_data = process_and_display_image(img, conf)

            st.image(img_pil, caption="Captured Image", use_column_width=True)
            st.json(labels)
            st.download_button(label="Download Image with Detections", data=img_pil.tobytes(),
                               file_name="captured_image_detected.jpg", mime="image/jpeg")

if __name__ == "__main__":
    main()
