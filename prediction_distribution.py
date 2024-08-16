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

# from online source and was modified to adhere to our model and dataset to determine predictions
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "model_CNN_bs15_lr1e-05_epoch144"
    model = CNNClassifier().to(device)  # Move model to the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    large_dataset = torchvision.datasets.ImageFolder(root="Large_Dataset", transform=preprocess)
    loader = torch.utils.data.DataLoader(large_dataset, batch_size=32, shuffle=True)

    accuracy = get_accuracy(model, loader, device)
    print(f"Test accuracy: {accuracy*100:.2f} %")

    folder_path = "Large_Dataset"  
    
    images = []
    predictions = []
    ground_truths = []


    for root, _, files in os.walk(folder_path):  
        for image_name in files:
            if image_name.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, image_name)
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)  

                # Perform inference
                with torch.no_grad():
                    out = model(image_tensor)

                prob = F.softmax(out, dim=1)

                predicted_class = torch.argmax(prob, dim=1).item()

                class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
                predicted_label = class_labels[predicted_class]

                ground_truth_label = os.path.basename(root) 

                images.append(image)
                predictions.append(predicted_label)
                ground_truths.append(ground_truth_label)

    n_images = len(images)
    n_cols = 7  
    n_rows = max((n_images + n_cols - 1) // n_cols, 1)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 3 * n_rows) 
    )

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(images[i])
            ax.set_title(f"Predicted: {predictions[i]}\nGround Truth: {ground_truths[i]}")
            ax.axis("off") 
            ax.set_aspect('auto', adjustable='box')
        else:
            ax.axis("off"

    plt.tight_layout()
    plt.show()
