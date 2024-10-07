import os
import json
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names from the class_indices.json file
class_indices_path = f"{working_dir}/class_indices.json"
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Create a list of class labels from the loaded class indices (keys are the indices, values are labels)
class_labels = [class_indices[str(i)] for i in range(len(class_indices))]

# Initialize video capture (0 is the default webcam; switch backend if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set video resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output_path = os.path.join(working_dir, 'output_video.mp4')  # Save video as 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))  # 20 FPS, 640x480 resolution

# Loop to capture frames from the webcam/video feed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to the input size expected by the model (e.g., 224x224)
    input_frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame: convert it to array and normalize it (as done during training)
    input_frame = img_to_array(input_frame)
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension (for model)
    input_frame = input_frame / 255.0  # Normalize pixel values

    # Predict the class of the frame
    prediction = model.predict(input_frame)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get class index
    confidence = np.max(prediction)  # Get confidence score

    # Get the corresponding class label from the class_labels list
    predicted_class_label = class_labels[predicted_class_index]

    # Display the result on the video frame
    label = f"{predicted_class_label} ({confidence * 100:.2f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the video file
    out.write(frame)

    # Show the frame with the prediction in a window
    cv2.imshow('Plant Disease Detection', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture, writer, and close the window
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path}")