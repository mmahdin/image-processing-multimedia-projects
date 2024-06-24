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


# Example: Capture an image from the webcam and classify it
captured_image_path = './cam/captured_image.jpg'
annotated_image_path = './cam/annotated_image.jpg'

# image = capture_image_from_webcam(save_path=captured_image_path)
# annotated_image, class_label = classify_image(image)
# annotated_image.save(annotated_image_path)  # Save the annotated image

# Example: Load an image from the disk and classify it
annotated_image_path = './cam_img/annotated_image.jpg'
image_path = './cam_img/images.jpg'
image = Image.open(image_path)
annotated_image, class_label = classify_image(image)
annotated_image.save(annotated_image_path)  # Save the annotated image
