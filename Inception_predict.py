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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define the model
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
model.load_state_dict(torch.load('inception.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Folder containing images for prediction
input_folder = '/test/'

# Lists to store ground truth and predicted labels
true_labels = []
predicted_labels = []

# Mapping of folder names to numeric labels
label_mapping = {'correct': 0, 'incorrect': 1}

# Iterate through the subfolders (class folders)
for class_folder in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_folder)

    if os.path.isdir(class_path):
        # Get the numeric label for the class
        true_label = label_mapping.get(class_folder, -1)  # default to -1 if not found
        if true_label != -1:
            # Iterate through the images in the class folder
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Extend true_labels for each image
                    true_labels.append(true_label)

                    # Load and preprocess the image
                    img_path = os.path.join(class_path, filename)
                    img = Image.open(img_path)
                    img_tensor = transform(img)
                    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add a batch dimension

                    # Make prediction
                    with torch.no_grad():
                        model.eval()
                        output = model(img_tensor)

                    # Get the predicted class
                    _, predicted_class = torch.max(output, 1)
                    predicted_labels.append(predicted_class.item())
                    
                    
print(f"true labels: {true_labels}")
print(f"predicted labels: {predicted_labels}")

# Derive class names from unique labels in true_labels
class_names = list(set(true_labels))

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_names)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


# Calculate the total number of samples
total_samples = np.sum(conf_matrix)

# Calculate the number of correctly classified samples (diagonal elements)
correctly_classified = np.trace(conf_matrix)

# Calculate the number of misclassified samples (off-diagonal elements)
misclassified = total_samples - correctly_classified

# Calculate the percentage of misclassified samples
misclassification_percentage = (misclassified / total_samples) * 100

print(f"Total samples: {total_samples}")
print(f"Correctly classified: {correctly_classified}")
print(f"Misclassified: {misclassified}")
print(f"Misclassification Percentage: {misclassification_percentage:.2f}%")
# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Save the plot as an image file (adjust the filename and format as needed)
plt.savefig('C:/Users/Incase/runs/detect/confusion_matrix_cardinal_health.png')
plt.show()
