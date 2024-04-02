# A general approach how to train a model using custom dataset

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Function to read and preprocess images
def preprocess_image(image_path, target_size=(28, 28)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size).flatten() / 255.0
    return image

# Load custom dataset
dataset_dir = "path/to/your/custom/dataset"
classes = sorted(os.listdir(dataset_dir))
images = []
labels = []

for class_id, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        images.append(preprocess_image(image_path))
        labels.append(class_id)

X = np.array(images)
Y = np.array(labels)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

#choose your classifier from models.py, import and run to get y_pred and accuracy
y_pred = "" # result[0], image classification function from models.py
accuracy = "" # result[1]
conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
