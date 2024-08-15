import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import time
import math
import os
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models
import torchvision
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=4):
        self.name = "CNN"
        super(CNNClassifier, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        # Freeze the layers except the final layer
        for param in self.vgg19.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer to match the number of classes
        num_ftrs = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.vgg19(x)
        return x

if __name__ == "__main__":
    root_str = 'combined_data\\'

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])
    dataset3_val = torchvision.datasets.ImageFolder(root=root_str + 'valid', transform=transform)
    dataset3_train = torchvision.datasets.ImageFolder(root=root_str + 'train', transform=transform)
    dataset3_test = torchvision.datasets.ImageFolder(root=root_str + 'test', transform=transform)

    val3_loader = torch.utils.data.DataLoader(dataset3_val, batch_size=64, num_workers=0, shuffle=True)
    train3_loader = torch.utils.data.DataLoader(dataset3_train, batch_size=64, num_workers=0, shuffle=True)
    test3_loader = torch.utils.data.DataLoader(dataset3_test, batch_size=64, num_workers=0, shuffle=True)

    def get_model_name(name, batch_size, learning_rate, epoch):
        """Generate a name for the model consisting of all the hyperparameter values

        Args:
            config: Configuration object containing the hyperparameters
        Returns:
            path: A string with the hyperparameter name and value concatenated
        """
        path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(
            name, batch_size, learning_rate, epoch
        )
        return path

    def get_accuracy(model, data_loader):

        correct = 0
        total = 0
        for imgs, labels in data_loader:

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            output = model(imgs)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.shape[0]
        return correct / total

    def plot_training_curve(path):
        """Plots the training curve for a model run, given the csv files
        containing the train/validation error/loss.

        Args:
            path: The base path of the csv files produced during training
        """
        import matplotlib.pyplot as plt

        train_acc = np.loadtxt("{}_train_acc.csv".format(path))
        val_acc = np.loadtxt("{}_val_acc.csv".format(path))
        train_loss = np.loadtxt("{}_train_loss.csv".format(path))
        val_loss = np.loadtxt("{}_val_loss.csv".format(path))
        plt.title("Train vs Validation Error")
        n = len(train_acc)  # number of epochs
        plt.plot(range(1, n + 1), train_acc, label="Train")
        plt.plot(range(1, n + 1), val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.show()
        plt.title("Train vs Validation Loss")
        plt.plot(range(1, n + 1), train_loss, label="Train")
        plt.plot(range(1, n + 1), val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.show()

    use_cuda = True

    def train(
        model,
        data1=dataset3_train,
        data2=dataset3_val,
        batch_size=64,
        num_workers=2,
        num_epochs=150,
        learning_rate=0.00008,
        gamma = 0.99
    ):

        torch.manual_seed(10)

        if use_cuda and torch.cuda.is_available():
            model.cuda()
            print("GPU is available")

        train_loader = torch.utils.data.DataLoader(
            data1, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            data2, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        # train_acc, val_acc = [], []
        train_acc = np.zeros(num_epochs)
        val_acc = np.zeros(num_epochs)

        # training
        print("Started Training")
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            for imgs, labels in train_loader:

                #############################################
                # To Enable GPU Usage
                if use_cuda and torch.cuda.is_available():
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                #############################################

                out = model(imgs)  # forward pass
                loss = criterion(out, labels)  # compute the total loss
                loss.backward()  # backward pass (compute parameter updates)
                optimizer.step()  # make the updates for each parameter
                optimizer.zero_grad()  # a clean up step for PyTorch

                # print("Finished Batch")

            print("Finished Epoch")
            model.eval()  # this ensures you are not usind dropout when calculating accuracy
            # save the current training information
            train_acc[epoch] = get_accuracy(
                model, train_loader
            )  # compute training accuracy
            val_acc[epoch] = get_accuracy(
                model, val_loader
            )  # compute validation accuracy
            # Save the current model (checkpoint) to a file
            model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)
            print(
                ("Epoch {}: Train Acc: {}, Validation Acc: {}").format(
                    epoch, train_acc[epoch], val_acc[epoch]
                )
            )
            scheduler.step()
        print("Finished Training")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

        epochs = np.arange(1, num_epochs + 1)

        # plotting

        plt.title("Accuracy Curves")
        plt.plot(epochs, train_acc, label="Dataset 1 (Usually Training Dataset)")
        plt.plot(epochs, val_acc, label="Dataset 2 (Usually Validation Dataset)")
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy")
        plt.legend(loc="best")
        plt.show()

        print("Final Training Accuracy: {}".format(train_acc[-1]))
        print("Final Validation Accuracy: {}".format(val_acc[-1]))

    test_model = CNNClassifier(num_classes=4)
    train(
        test_model,
        data1=dataset3_train,
        batch_size=25,
        num_epochs=250,
        learning_rate=0.000001,
    )
