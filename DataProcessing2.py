# modified code from an online source to sort data and create a combined dataset
# mount Google Drive
from google.colab import drive

drive.mount("/content/gdrive")

# import necessary libraries
import os
import shutil
import random

# set paths
dataset3_val = "/content/gdrive/MyDrive/APS360/Dataset_3_classes/valid"
dataset3_tra = "/content/gdrive/MyDrive/APS360/Dataset_3_classes/train"
dataset3_tes = "/content/gdrive/MyDrive/APS360/Dataset_3_classes/test"

new_base_dir = "/content/gdrive/MyDrive/APS360/Dataset_3_classes_redistributed"
new_train_dir = os.path.join(new_base_dir, "train")
new_val_dir = os.path.join(new_base_dir, "valid")
new_test_dir = os.path.join(new_base_dir, "test")

# create new directories
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_val_dir, exist_ok=True)
os.makedirs(new_test_dir, exist_ok=True)

classes = ["glioma", "meningioma", "notumor", "pituitary"]

for cls in classes:
    os.makedirs(os.path.join(new_train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, cls), exist_ok=True)


# collect all files, shuffle, and distribute
def collect_files(directory):
    files = {}
    for cls in classes:
        cls_dir = os.path.join(directory, cls)
        cls_files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f))
        ]
        files[cls] = cls_files
    return files


all_files = {cls: [] for cls in classes}
for cls in classes:
    all_files[cls].extend(collect_files(dataset3_tra).get(cls, []))
    all_files[cls].extend(collect_files(dataset3_val).get(cls, []))
    all_files[cls].extend(collect_files(dataset3_tes).get(cls, []))
    random.shuffle(all_files[cls])

# define split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1


def copy_files(files, destination, count):
    copied_files = []
    for f in files[:count]:
        cls = os.path.basename(os.path.dirname(f))
        dest_dir = os.path.join(destination, cls)
        shutil.copy(f, dest_dir)
        copied_files.append(f)
    return copied_files


for cls in classes:
    total_files = len(all_files[cls])
    train_count = int(total_files * train_split)
    val_count = int(total_files * val_split)
    test_count = total_files - train_count - val_count

    train_files = copy_files(all_files[cls], new_train_dir, train_count)
    val_files = copy_files(all_files[cls][train_count:], new_val_dir, val_count)
    test_files = copy_files(
        all_files[cls][train_count + val_count :], new_test_dir, test_count
    )

    print(
        f"{cls} - Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}"
    )


# define the count_files function to return a dictionary with counts per class
def count_files(directory):
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(directory, cls)
        counts[cls] = len(
            [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        )
    return counts


# print the number of files for each class in train, valid, and test folders
train_counts = count_files(new_train_dir)
val_counts = count_files(new_val_dir)
test_counts = count_files(new_test_dir)

total_train_files = sum(train_counts.values())
total_val_files = sum(val_counts.values())
total_test_files = sum(test_counts.values())

print("\nTraining files per class:")
for cls, count in train_counts.items():
    percentage = (count / total_train_files) * 100
    print(f"{cls}: {count} ({percentage:.2f}%)")

print("\nValidation files per class:")
for cls, count in val_counts.items():
    percentage = (count / total_val_files) * 100
    print(f"{cls}: {count} ({percentage:.2f}%)")

print("\nTest files per class:")
for cls, count in test_counts.items():
    percentage = (count / total_test_files) * 100
    print(f"{cls}: {count} ({percentage:.2f}%)")
