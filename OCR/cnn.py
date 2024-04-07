# to try things out
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to resize and preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (64, 64))
    # Normalizing pixel values to range [0, 1]
    normalized_image = resized_image / 255.0
    return normalized_image

# Load the dataset and preprocess images
def load_dataset(train_dir):
    features = []
    labels = []
    folders = os.listdir(train_dir)
    for label in folders:
        folder_path = os.path.join(train_dir, label)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)
            preprocessed_img = preprocess_image(img)
            features.append(preprocessed_img)
            labels.append(int(label)-1)  # Adjusting labels to start from 0
    return np.array(features), np.array(labels)

# Path to the directory containing the image folders
train_dir = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample images"

# Load dataset
X, y = load_dataset(train_dir)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train.reshape(-1, 64*64*3), y_train)

# Reshape back to image shape
X_train_resampled = X_train_resampled.reshape(-1, 64, 64, 3)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(41, activation='softmax')  # 41 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Predict on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Evaluate the classifier
print('Accuracy {:.2f}'.format(test_acc))
conf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(threshold=np.inf)
print("Confusion Matrix:")
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Save the model
save_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\OCR\cnn_model.h5'
model.save(save_path)
print("Model saved to:", save_path)

# Load the model
loaded_model = tf.keras.models.load_model(save_path)

# Now classify a single input image
image_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample images\1\im__22.png"
img = cv2.imread(image_path)
preprocessed_img = preprocess_image(img)
predicted_label = np.argmax(loaded_model.predict(np.expand_dims(preprocessed_img, axis=0)), axis=-1)
print("Predicted label:", predicted_label[0]+1)  # Adjusting label to match original numbering


