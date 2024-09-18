"""

@author: Jinal Shah

This script is used to take in 
video feed from the laptop camera and
then break it down into frames and feed
it through the model to get the predictions.

"""
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import v2


model = nn.Sequential(
    nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.BatchNorm2d(64),
    nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.BatchNorm2d(128),
    nn.Dropout(p=0.5),
    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.BatchNorm2d(512),
    nn.Dropout(p=0.5),
    nn.Flatten(),
    nn.Linear(131072,500),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(500,10)
)

# Load the model weights
model.load_state_dict(torch.load('../models/cnn_3.pth'))
model.eval()

# Define the transformation to be applied to each frame
transformation_pipeline = v2.Compose([
        v2.ToImage(),
        v2.Resize([128,128]),
        v2.ToDtype(torch.float32, scale=True),
])

# Start capturing video from the laptop's camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the captured frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transformations
    frame_transformed = transformation_pipeline(frame_rgb)

    # Add a batch dimension
    frame_transformed = frame_transformed.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        predictions = model(frame_transformed)

    probs = torch.nn.functional.softmax(predictions, dim=1)
    max_prob, pred_label_idx = torch.max(probs, 1)
    labels = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']
    pred_label = labels[pred_label_idx.item()]

    # Display the frame
    cv2.putText(frame, f"Label: {pred_label}, Probability: {max_prob.item():.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()