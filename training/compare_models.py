import os
import numpy as np
import ast
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

IMG_SIZE = 224
TEST_PATH = "dataset/Indian currency dataset v1/test"

# Paths to the two models
MODELS = {
    "MobileNetV2": "currency_model.h5",
    "ResNet50": "currency_resnet_model.h5"
}

# Verify models exist
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"Error: {name} model file '{path}' not found. Please run training first.")
        exit(1)

# Load class indices
with open("class_indices.txt", "r") as f:
    class_indices = ast.literal_eval(f.read())
class_labels_mapped = {v: k for k, v in class_indices.items()}
class_names = [class_labels_mapped[i] for i in range(len(class_indices))]

results = {}

for model_name, model_path in MODELS.items():
    print(f"\nEvaluating Model: {model_name}...")
    model = load_model(model_path)
    
    true_labels = []
    predicted_labels = []

    for img_name in os.listdir(TEST_PATH):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            true_label = img_name.split("_")[0]
            if true_label not in class_indices:
                print(f"Warning: Skipping {img_name} with unexpected label '{true_label}'")
                continue

            img_path = os.path.join(TEST_PATH, img_name)

            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # MODEL-SPECIFIC PREPROCESSING
            if model_name == "MobileNetV2":
                img_array /= 255.0
            elif model_name == "ResNet50":
                import tensorflow as tf
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            prediction = model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(prediction)
            
            true_labels.append(true_label)
            predicted_labels.append(class_labels_mapped[predicted_class_index])

    # Metrics
    results[model_name] = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, labels=class_names, target_names=class_names)
    
    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {results[model_name]:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save Classification Report to text file
    report_filename = f"{model_name.lower()}_report.txt"
    with open(report_filename, "w") as f:
        f.write(report)
    print(f"Classification report saved as {report_filename}")

    # Generate Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}')
    cm_filename = f"{model_name.lower()}_confusion_matrix.png"
    plt.savefig(cm_filename)
    print(f"Confusion matrix saved as {cm_filename}")
    plt.close()

# Plot comparison bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(results.keys(), results.values(), color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Model Comparison: MobileNetV2 vs ResNet50')
plt.ylim(0, 1.1)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
plt.savefig('comparison_results.png')
plt.show()

print("\nFinal comparison bar chart saved as comparison_results.png")
