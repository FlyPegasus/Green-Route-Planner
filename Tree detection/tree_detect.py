'''
dependencies:
pip install tensorflow opencv-python scikit-learn
pip install keras

'''

import numpy as np
import cv2
import tensorflow as tf

# Load the pretrained Mask R-CNN model
model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# Print the model summary to see the layers and parameters
model.summary()


# Read the image
image_path = 'satimg.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to the input size of the model
input_size = (224, 224)  # Example size, change based on the model requirement
image_resized = cv2.resize(image_rgb, input_size)

# Normalize the image
image_normalized = image_resized / 255.0

# Expand dimensions to match the input shape of the model
input_image = np.expand_dims(image_normalized, axis=0)


# Perform inference
predictions = model.predict(input_image)

# Process the predictions to extract bounding boxes and class labels
# (This part will vary based on the specific model and output format)


def get_coordinates(predictions, confidence_threshold=0.5):
    coordinates = []
    for pred in predictions:
        # Extract the bounding box coordinates and class label
        (startX, startY, endX, endY) = pred['box']
        confidence = pred['confidence']

        # Filter out weak detections
        if confidence > confidence_threshold:
            coordinates.append((startX, startY, endX, endY))
    return coordinates


coordinates = get_coordinates(predictions)

# Print the coordinates of detected trees
for coord in coordinates:
    print(f"Tree detected at: {coord}")


def draw_bounding_boxes(image, coordinates):
    for (startX, startY, endX, endY) in coordinates:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return image


# Draw bounding boxes on the original image
image_with_boxes = draw_bounding_boxes(image, coordinates)

# Display the image
cv2.imshow('Detected Trees', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
