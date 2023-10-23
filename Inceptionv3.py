# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 08:51:53 2023

@author: Incase
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# Assume your dataset is structured as follows:
# root/
#   |- correct/
#   |- incorrect/
train_dataset = datasets.ImageFolder(root='C:/Users/Incase/Documents/Inception_pen_1/train', transform=transform)
test_dataset = datasets.ImageFolder(root='C:/Users/Incase/Documents/Inception_pen_1/test', transform=transform)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Assuming train_loader is your DataLoader object
class_names = train_loader.dataset.classes
print("Class Names:", class_names)
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), 'C:/Users/Incase/Downloads/ultralytics-main/pen_inception_1.pth')   
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy}")

########## prediction on new image ##############

# Load the trained model
model_path = 'C:/Users/Incase/Downloads/ultralytics-main/pen_inception_2.pth'
model = torch.load(model_path)  # Note: Use torch.load() for models saved with torch.save()

# Set the model to evaluation mode
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Load and preprocess the new image
image_path = '/path/to/your/new/image.jpg'
img = Image.open(image_path)
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

# Make prediction
with torch.no_grad():
    model.eval()
    output = model(img_tensor)

# Get the predicted class
_, predicted_class = torch.max(output, 1)

print(f"Predicted Class: {predicted_class.item()}")
