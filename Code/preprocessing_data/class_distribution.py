import os
import matplotlib.pyplot as plt

def analyze_class_distribution(directory):
    """
    Analyze and visualize class distribution in the dataset
    """
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))

    # Visualize class distribution
    plt.figure(figsize=(15, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return class_counts

class_distribution = analyze_class_distribution(original_dataset_dir)
