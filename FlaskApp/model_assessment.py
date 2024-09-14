import os
import cv2
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load the trained model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2  # 2 classes: real (0) and fake (1)
)
model.load_state_dict(torch.load("best_model_epoch_X.pth"))  # Replace X with the appropriate epoch number
model.eval()  # Set model to evaluation mode

# Initialize ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Define the transform (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to predict whether a video is fake or real
def is_video_fake(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frame_predictions = []
    
    # Read video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (OpenCV reads frames in BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame
        frame_tensor = transform(frame)

        # Pass through feature extractor
        inputs = feature_extractor(images=[frame], return_tensors="pt").pixel_values.squeeze(1)
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
        
        # Store prediction for the frame (0: real, 1: fake)
        frame_predictions.append(predicted.item())
    
    cap.release()

    # Majority voting across all frames in the video
    fake_votes = sum(frame_predictions)
    total_frames = len(frame_predictions)
    
    # Determine if the video is fake based on majority of frame predictions
    if fake_votes > total_frames // 2:
        return "Fake"
    else:
        return "Real"

# # Example usage
# video_file = "path_to_video.mp4"  # Path to the video you want to check
# result = is_video_fake(video_file)
# print(f"The video is predicted to be: {result}")
