import os
import random
import shutil

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories for train, val, and test
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Process each class
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Get all files in the class directory
    files = os.listdir(class_dir)
    random.shuffle(files)

    # Compute split sizes
    total_files = len(files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # Split files
    train_files = files[:train_size]
    val_files = files[train_size : train_size + val_size]
    test_files = files[train_size + val_size :]

    # Copy files to the respective directories
    for split, split_files in zip(
        ["train", "val", "test"], [train_files, val_files, test_files]
    ):
        split_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for file in split_files:
            src_file = os.path.join(class_dir, file)
            dest_file = os.path.join(split_dir, file)
            shutil.copy(src_file, dest_file)

print("Dataset split completed!")
