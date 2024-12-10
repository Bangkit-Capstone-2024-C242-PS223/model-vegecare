import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def evaluate_model_performance(model, test_ds, classes):
    # Collect true labels and predictions
    true_labels = []
    predictions = []

    # Iterate through test dataset to get true labels and predictions
    for images, labels in test_ds:
        # Get class indices from one-hot encoded labels
        true_batch_labels = np.argmax(labels.numpy(), axis=1)
        true_labels.extend(true_batch_labels)

        # Get model predictions
        batch_predictions = model.predict(images)
        predictions.extend(np.argmax(batch_predictions, axis=1))

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # 1. Detailed Classification Report
    print("Detailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=classes))

    # 2. Confusion Matrix Visualization
    plt.figure(figsize=(20, 15))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # 3. Per-class Accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(classes):
        class_total = np.sum(true_labels == i)
        class_correct = np.sum((true_labels == i) & (predictions == i))
        accuracy = class_correct / class_total if class_total > 0 else 0
        per_class_accuracy[class_name] = {
            "total_samples": class_total,
            "correct_predictions": class_correct,
            "accuracy": accuracy,
        }

    # Print Per-class Accuracy
    print("\nPer-Class Accuracy:")
    for class_name, stats in per_class_accuracy.items():
        print(f"{class_name}:")
        print(f"  Total Samples: {stats['total_samples']}")
        print(f"  Correct Predictions: {stats['correct_predictions']}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")

    return true_labels, predictions, per_class_accuracy


# Use the function after training
true_labels, predictions, per_class_accuracy = evaluate_model_performance(
    model, test_ds, classes
)
def calculate_advanced_metrics(true_labels, predictions):
    # Macro average (treats all classes equally)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro"
    )

    # Weighted average (considers class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(true_labels, predictions, average="weighted")
    )

    print("\nAdvanced Metrics:")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")


# Calculate after getting predictions
calculate_advanced_metrics(true_labels, predictions)
