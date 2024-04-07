# logistic regression

import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from skimage.feature import hog
import joblib

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Preprocess the image
    img_res = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    hog_img = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    return hog_img

# Load the dataset and extract features
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
            hog_features = extract_hog_features(img)
            features.append(hog_features)
            labels.append(label)
    return features, labels

# Train logistic regression model
def train_model(X_train, y_train):
    logistic_reg = LogisticRegression()
    clf = logistic_reg.fit(X_train, y_train)
    return clf

# Path to the directory containing the image folders
train_dir = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample images"

# List of folder names, each containing images of a handwritten character
folders = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
    "41"
]

# Load dataset
X, y = load_dataset(train_dir)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Train logistic regression model
clf = train_model(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print('Accuracy {:.2f}'.format(clf.score(X_test, y_test)))
conf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(threshold=np.inf)
print("Confusion Matrix:")
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)



# Load the trained classifier from the file
save_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\final project\model.joblib'

try:
    loaded_clf = joblib.load(save_path)
    print("Classifier loaded successfully.")
except Exception as e:
    print("Error loading classifier:", e)
# Save the trained classifier to a file
joblib.dump(clf, save_path)
print("Classifier saved to:", save_path)

# Now classify a single input image
image_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample images\1\im__22.png"
img = cv2.imread(image_path)
hog_features = extract_hog_features(img)
predicted_label = loaded_clf.predict([hog_features])
print("Predicted label:", predicted_label[0])
