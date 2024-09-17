from flask import Flask, render_template, request, jsonify, url_for
import os
import shutil
from datetime import datetime
now = datetime.now()

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FRAMES_FOLDER = 'static/frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_FRAMES_FOLDER'] = EXTRACTED_FRAMES_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

# Load the pre-trained model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    ignore_mismatched_sizes=True,
    num_labels=2  # 2 classes: real (0) and fake (1)
)
model.load_state_dict(torch.load("./Ml-Models/best_model_epoch_2.pth", weights_only=True))  # Replace X with the correct epoch number
model.eval()

# Initialize ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Define the transform (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Function to process and assess the video
def process_video(video_path, output_frames_dir):
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = 0
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and apply transformations
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb)

        # Process through the feature extractor and model
        inputs = feature_extractor(images=[frame_rgb], return_tensors="pt").pixel_values.squeeze(1)
        with torch.no_grad():
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)

        frame_predictions.append(predicted.item())

        # Save the frame as an image
        frame_filename = os.path.join(output_frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        frame_count += 1

    cap.release()

    # Assess video based on frame predictions (majority voting)
    fake_votes = sum(frame_predictions)
    total_frames = len(frame_predictions)
    result = "Fake" if fake_votes > total_frames // 2 else "Real"
    
    return result, frame_count

def delete_all_files_in_folder():
    if os.path.exists(EXTRACTED_FRAMES_FOLDER):
        shutil.rmtree(EXTRACTED_FRAMES_FOLDER)
        os.makedirs(EXTRACTED_FRAMES_FOLDER)
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    delete_all_files_in_folder()
    return render_template('index.html', year = now.year)

# Define the REST API endpoint for uploading videos
@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Process the video and get the result
        output_frames_dir = os.path.join(app.config['EXTRACTED_FRAMES_FOLDER'], filename.split('.')[0])
        result, frame_count = process_video(video_path, output_frames_dir)
        
        return jsonify({
            'result': result,
            'total_frames_processed': frame_count,
            'frame_directory': output_frames_dir
        })

    return jsonify({'error': 'File extension not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
