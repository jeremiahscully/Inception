# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:55:31 2023

@author: Incase
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

class CustomInception(nn.Module):
    def __init__(self, num_classes):
        super(CustomInception, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        
        # Change the output layer to match the number of classes in your dataset
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inception(x)

# Number of classes in your dataset (e.g., 2 for 'correct' and 'incorrect')
num_classes = 2

# Create the model
model = CustomInception(num_classes).to(device)

# Load the trained model state_dict
model.load_state_dict(torch.load('C:/Users/Incase/Downloads/ultralytics-main/pen_inception_2.pth', map_location=device))

# Set the model to evaluation mode
model.eval()
# Initialize counters
total_samples = 0
correct_predictions = 0
wrong_predictions = 0
# Folder containing mixed images
mixed_folder = 'C:/Users/Incase/Documents/Inception_pen_1/mixed/'

# Iterate through the images in the folder
for filename in os.listdir(mixed_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load and preprocess the image
        img_path = os.path.join(mixed_folder, filename)
        img = Image.open(img_path)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add a batch dimension

        # Make prediction
        with torch.no_grad():
            model.eval()
            output = model(img_tensor)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)

        # Print the result
        print(f"Image: {filename}, Predicted Class: {predicted_class.item()}")
        
# Extract the numeric part of the filename using regular expressions
        numeric_part = re.search(r'\d+', filename)
        true_label = int(numeric_part.group()) if numeric_part is not None else None


        # Print information
        print(f"File name: {filename}, True Label: {true_label}, Predicted Class: {predicted_class.item()}")

        # Update counters
        total_samples += 1
        if predicted_class.item()==0:
            correct_predictions += 1
        else:
            wrong_predictions += 1
results=(correct_predictions/total_samples)*100
# Print summary
print(f"Total Samples: {total_samples}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Wrong Predictions: {wrong_predictions}")
print(f"percentatge: {results}")





