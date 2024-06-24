import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import cv2
import sys
import numpy as np
from PySide6.QtCore import QThread, Signal
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


capture_image_flg = 0  

# CSS-like button styles for various functionalities
button_style_send = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        min-width: 50px;
        font-size: 24px;
        border-image: url(./pngs/upload.png);
    }
    
    QPushButton:pressed {
        border-image: url(./pngs/uploadp.png);
    }
"""


class WebcamThread(QThread):
    # Define a signal to emit the webcam frame
    change_pixmap_signal = Signal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)  # Initialize the webcam capture
        while True:
            ret, frame = cap.read()  # Read a frame from the webcam
            if ret:  # Check if the frame is successfully captured
                # Emit the signal with the captured frame
                self.change_pixmap_signal.emit(frame)
            else:
                pass  # If frame capture fails, do nothing to handle the error


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set size of window
        self.setWindowTitle("Image Classification")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(1050, 630)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set background color for the window
        self.back_clor = "f2f5ff"
        self.setStyleSheet(f"background-color: #{self.back_clor};")

        # Create GUI layout
        self.create_layout()

        # Create and start webcam thread
        self.webcam_thread = WebcamThread()
        self.webcam_thread.change_pixmap_signal.connect(self.update_image)
        self.webcam_thread.start()
        
    def create_layout(self):
        # Create main layout for central widget
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        ################################### Image layout ###################################
        # Create widgets for displaying images and buttons
        image_widget = QWidget(self)
        left_vertical_widget = QWidget(image_widget)
        right_vertical_widget = QWidget(image_widget)

        # Create a horizontal layout for arranging image widgets
        image_layout = QHBoxLayout()
        image_layout.setObjectName("imageLayout")

        # Set the layout for the image widget
        image_widget.setLayout(image_layout)
        # Add the image widget to the main layout with a stretch factor of 10
        main_layout.addWidget(image_widget, 10)

        # Create QLabel instances for displaying images
        self.left_image_label = QLabel(left_vertical_widget)
        self.left_image_label.resize(500, 500)
        self.left_image_label.move(50, 0)

        # Create QLabel instance for displaying images on the right side
        self.right_top_image_label = QLabel(right_vertical_widget)
        self.right_top_image_label.resize(500, 500)

        # Add left and right widgets to the image layout
        image_layout.addWidget(left_vertical_widget)
        image_layout.addWidget(right_vertical_widget)

        ################################### Send layout ###################################
        # Create a QPushButton instance for sending functionalities
        self.camera = ClickableLabel(self)
        # self.camera.setStyleSheet(button_style_send)
        self.camera.setStyleSheet("border-radius: 50%;")
        self.camera.move(420, 520)
        self.camera.resize(100, 100)
        self.camera.clicked.connect(self.capture_image)
        # self.camera.clicked.connect(self.send_message)

        loadImg = QPushButton(self)
        loadImg.setStyleSheet(button_style_send)
        loadImg.move(530, 540)
        loadImg.resize(70, 70)
        loadImg.clicked.connect(self.upload_image)

    def update_image(self, cv_img):
        global capture_image_flg

        # capture image and save it
        if capture_image_flg == 1:
            cv2.imwrite("./cam/myimg.png", cv_img)
            self.process_image("./cam/myimg.png")
            self.update_left_image(cv_img)
            capture_image_flg = 0
            cv_img = np.ones_like(cv_img)

        # Resize the image to fit the QLabel
        cv_img = cv2.resize(cv_img, (100, 100))

        # Create a circular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 50, 1, -1)

        # Apply the mask to the image
        # cv_img = cv2.bitwise_and(cv_img, cv_img, mask=mask)

        # Convert mask to 3 channels
        mask_3ch = cv2.merge([mask, mask, mask])

        # Create a background with the specified color
        bg_color = np.array([int(self.back_clor[4:6],16), int(self.back_clor[2:4],16), int(self.back_clor[0:2],16)], dtype=np.uint8)  # color #f2f5ff
        background = np.full(cv_img.shape, bg_color, dtype=np.uint8)

        # Combine the image and the background using the mask
        cv_img = np.where(mask_3ch == 1, cv_img, background)

        # Convert the OpenCV image to a Qt image
        qt_img = QImage(
            cv_img.data, cv_img.shape[1], cv_img.shape[0], QImage.Format_RGB888).rgbSwapped()
        
        # Create a QPixmap from the Qt image and set it to the QLabel
        pixmap = QPixmap.fromImage(qt_img)

        # Create a new QPixmap to draw the border
        final_pixmap = QPixmap(100, 100)
        final_pixmap.fill(Qt.transparent)  # Ensure the pixmap is transparent

        # Create a QPainter to draw on the pixmap
        painter = QPainter(final_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the image onto the final pixmap
        painter.drawPixmap(0, 0, pixmap)

        # Set the pen to draw the circle border
        pen = QPen(QColor('#8177ff'))
        pen.setWidth(3)  # Set the width of the border
        painter.setPen(pen)

        # Draw the circle border
        painter.drawEllipse(1, 1, 98, 98)  # Adjust to fit within the pixmap

        # End the painting
        painter.end()

        self.camera.setPixmap(final_pixmap)
        
    def capture_image(self):
        """
        Handle the functionality for capturing an image.
        """
        global capture_image_flg
        capture_image_flg = 1
        pass

    def update_left_image(self, cv_img):
        """
        Update the image displayed on the left side of the window.

        Args:
            cv_img (numpy.ndarray): Image data in OpenCV format.

        Returns:
            None
        """
        # Extract height, width, and channels from the image
        height, width, channel = cv_img.shape
        # Calculate bytes per line
        bytesPerLine = 3 * width
        # Convert the OpenCV image to a Qt image
        qt_img = QImage(cv_img.data, width, height,
                        bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        # Create a QPixmap from the Qt image and set it to the QLabel
        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
        self.left_image_label.setPixmap(pixmap)
        self.left_image_label.move(0, 0)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
            self.left_image_label.setPixmap(pixmap)
            self.process_image(file_name)

    def process_image(self, path):
        annotated_image_path = './cam/annotated_image.jpg'
        image = Image.open(path)
        annotated_image, class_label = classify_image(image)
        annotated_image.save(annotated_image_path)  # Save the annotated image
        pixmap = QPixmap(annotated_image_path)
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
        self.right_top_image_label.setPixmap(pixmap)
        
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def capture_image_from_webcam(save_path='captured_image.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Failed to capture image")
    
    # Convert the image from BGR (OpenCV format) to RGB (PIL format)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    
    # Save the captured image
    image.save(save_path)
    return image

def annotate_image_with_class(image, class_label, save_path='./cam/annotated_image.jpg'):
    # Convert to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Define the font and location for the label
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (10, 30)
    font_scale = 1
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2
    
    # Put the class label on the image
    annotated_image = cv2.putText(image_cv, class_label, location, font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Convert back to PIL format and save the annotated image
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    annotated_image.save(save_path)
    
    return annotated_image

def predict_image_class(image):
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    _, predicted = outputs.max(1)
    
    # Load ImageNet class names
    class_file = 'imagenet_classes.txt'
    if not os.path.exists(class_file):
        raise FileNotFoundError(f"{class_file} not found. Please download it and place it in the working directory.")
    
    with open(class_file) as f:
        class_names = [line.strip() for line in f.readlines()]
    
    predicted_class = class_names[predicted.item()]
    return predicted_class

def classify_image(image):
    class_label = predict_image_class(image)
    annotated_image = annotate_image_with_class(image, class_label)
    return annotated_image, class_label


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())