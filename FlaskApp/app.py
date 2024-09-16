from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification

from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/upload-video', methods=['POST'])
def extract_video():
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)