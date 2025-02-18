import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO
model = YOLO("model/best.pt")

# Load image 
image = cv2.imread('images\lembar jawaban manual.jpg')

# Detect objects
results = model([image])

# Cropped image
cropped_images = []

for result in results:
    for box in result.boxes:
        # Draw bounding box
        #cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) ), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 1)
        
        # Crop image
        cropped_images.append(image[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])])
        
        # Draw class
        #cv2.putText(image, f"{result.names[int(box.cls[0])]}", (int(box.xyxy[0][0]), int(box.xyxy[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# Display
cv2.imshow('image', image)

# Resize all cropped images to the same height
if cropped_images:
    max_height = max(img.shape[0] for img in cropped_images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in cropped_images]
    combined_image = np.hstack(resized_images)
    cv2.imshow('Cropped Images', combined_image)

# Do Prepossing cropped images
# Convert RGB to gray
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in cropped_images]

# Do thresholding  
threshold_images = [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for img in gray_images]

# Display
if threshold_images:
    max_height = max(img.shape[0] for img in threshold_images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in threshold_images]
    combined_image = np.hstack(resized_images)
    cv2.imshow('Threshold Images', combined_image)


cv2.waitKey(0)
cv2.destroyAllWindows()