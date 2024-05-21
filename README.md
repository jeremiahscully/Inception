
# InceptionV3 Image Classification

This repository contains an implementation of the InceptionV3 model using PyTorch. The model is used for image classification tasks, specifically classifying images into two categories: "correct" and "incorrect".

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The InceptionV3 model is a deep convolutional neural network architecture developed by Google. It is designed for image classification tasks and has achieved state-of-the-art results on various benchmarks. This implementation trains the InceptionV3 model on a custom dataset and evaluates its performance.

## Requirements

- Python 3.6 or higher
- PyTorch
- torchvision
- PIL (Pillow)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jeremiahscully/inceptionv3-image-classification.git
    cd inceptionv3-image-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install torch torchvision pillow
    ```

## Usage

### Training

To train the InceptionV3 model, organize your dataset as follows:
```
root/
  |- train/
      |- correct/
      |- incorrect/
  |- test/
      |- correct/
      |- incorrect/
```

Then, run the training script:
```bash
python Inceptionv3.py
```
This script will train the model for 50 epochs and save the trained model to `inception_1.pth`.

### Evaluation

The training script also includes evaluation of the model on the test dataset after training. The accuracy of the model will be printed to the console.

### Prediction

To make predictions on new images, ensure you have a trained model saved as `inception_1.pth`. Update the path to the new image in the script and run the prediction section of the code:

```python
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
```

## Dataset

The dataset should be structured in a specific format with separate folders for training and testing data, and subfolders for each class ("correct" and "incorrect"). You can use any image dataset that fits this structure.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

