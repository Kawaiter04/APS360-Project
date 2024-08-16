import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import time
import os
import torchvision.models
from PIL import Image

# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        self.name = "CNN"
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.bn1 = nn.BatchNorm2d(5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 10)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 15, 15)
        self.bn3 = nn.BatchNorm2d(15)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4860, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 4860)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define the function to calculate accuracy
def get_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.shape[0]
    return correct / total

# Main function
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = "model_CNN_bs15_lr1e-05_epoch144"
    model = CNNClassifier().to(device)  # Move model to the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    large_dataset = torchvision.datasets.ImageFolder(root="Large_Dataset", transform=preprocess)
    loader = torch.utils.data.DataLoader(large_dataset, batch_size=32, shuffle=True)

    # Calculate and print the test accuracy
    accuracy = get_accuracy(model, loader, device)
    print(f"Test accuracy: {accuracy*100:.2f} %")

    folder_path = "Large_Dataset"  # Replace with your folder path

    # List to store images, predictions, and ground truths
    images = []
    predictions = []
    ground_truths = []

    # Loop through all the images in the folder structure
    for root, _, files in os.walk(folder_path):  # Walk through subfolders
        for image_name in files:
            if image_name.endswith((".jpg", ".jpeg", ".png")):
                # Load and preprocess the image
                image_path = os.path.join(root, image_name)
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)  # Move tensor to device

                # Perform inference
                with torch.no_grad():
                    out = model(image_tensor)

                # Convert the output to probabilities
                prob = F.softmax(out, dim=1)

                # Determine the predicted class
                predicted_class = torch.argmax(prob, dim=1).item()

                # Define the class labels
                class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
                predicted_label = class_labels[predicted_class]

                # Extract ground truth label from the folder name
                ground_truth_label = os.path.basename(root)  # Folder name as ground truth

                # Append image, prediction, and ground truth to the lists
                images.append(image)
                predictions.append(predicted_label)
                ground_truths.append(ground_truth_label)

    # Plot the images with their predictions and ground truths
    n_images = len(images)
    n_cols = 7  # Number of columns in the grid (set to 4 here)
    n_rows = max((n_images + n_cols - 1) // n_cols, 1)  # Calculate the number of rows, ensuring it's at least 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 3 * n_rows)  # Adjust figure size to make images smaller
    )

    # Loop through each subplot and display the image, prediction, and ground truth
    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(images[i])
            ax.set_title(f"Predicted: {predictions[i]}\nGround Truth: {ground_truths[i]}")
            ax.axis("off")  # Hide axes
            # Set the aspect ratio and make sure images fit well
            ax.set_aspect('auto', adjustable='box')
        else:
            ax.axis("off")  # Hide any extra empty subplots

    plt.tight_layout()
    plt.show()
