import sys
import cv2
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QVBoxLayout, QWidget, QFileDialog, QSlider
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from PIL import Image

class LungDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Abnormality Detection")
        self.setGeometry(100, 100, 800, 600)

        # Tambahkan gaya QSS
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;  /* Warna latar belakang */
            }
            QLabel {
                font-size: 18px;  /* Ukuran font */
                color: #333;  /* Warna teks */
                padding: 10px;
            }
            QPushButton {
                background-color: #4CAF50;  /* Warna latar belakang tombol */
                color: white;  /* Warna teks tombol */
                border: none;
                padding: 10px;
                border-radius: 5px;  /* Sudut tombol membulat */
                font-size: 16px;  /* Ukuran font tombol */
            }
            QPushButton:hover {
                background-color: #45a049;  /* Warna latar belakang tombol saat hover */
            }
            QSlider {
                background-color: #e0e0e0;  /* Warna latar belakang slider */
            }
            QSlider::groove:horizontal {
                background: #c0c0c0;  /* Warna groove slider */
                height: 10px;  /* Tinggi groove slider */
            }
            QSlider::handle:horizontal {
                background: #4CAF50;  /* Warna handle slider */
                border: 2px solid #333;  /* Border handle slider */
                width: 20px;  /* Lebar handle slider */
                margin-top: -5px;  /* Mengatur posisi handle slider */
                margin-bottom: -5px;
            }
        """)

        # Model YOLOv8
        self.model = YOLO("best14.pt")

        # Komponen UI
        self.image_label = QLabel("Upload or Capture an Image", self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(30)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(self.update_confidence)

        self.conf_label = QLabel("Confidence Threshold: 0.30", self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.capture_button)
        self.layout.addWidget(self.conf_label)
        self.layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.current_image = None

    def update_confidence(self):
        value = self.slider.value() / 100
        self.conf_label.setText(f"Confidence Threshold: {value:.2f}")

    def upload_image(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.display_image(path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cap.release()
            cv2.imwrite("captured_image.jpg", frame)
            self.display_image("captured_image.jpg")

    def display_image(self, path):
        self.current_image = cv2.imread(path)
        detections = self.detect_objects(self.current_image)

        annotated_image = self.annotate_image(self.current_image, detections)
        qimage = QImage(annotated_image.data, annotated_image.shape[1], 
                        annotated_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), 
                                                 self.image_label.height(), Qt.KeepAspectRatio))

    def detect_objects(self, image):
        results = self.model.predict(image)[0]
        confidence_threshold = self.slider.value() / 100
        detections = [(box, conf, class_id) for box, conf, class_id in 
                      zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls) 
                      if conf > confidence_threshold]
        return detections

    def annotate_image(self, image, detections):
        for box, conf, class_id in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(class_id)]} ({conf:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def save_detections(self, detections, file_name="detections.json"):
        detections_data = [
            {"box": box.tolist(), "confidence": float(conf), "class_id": int(class_id)}
            for box, conf, class_id in detections
        ]
        with open(file_name, "w") as f:
            json.dump(detections_data, f)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LungDetectionApp()
    window.show()
    sys.exit(app.exec_())
