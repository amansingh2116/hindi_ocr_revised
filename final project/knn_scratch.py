import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
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

# Define KNN function
def knn(X_train, y_train, X_test, k=5):
    y_pred = []
    for test_sample in X_test:
        distances = [np.linalg.norm(train_sample - test_sample) for train_sample in X_train]
        nearest_neighbors = np.argsort(distances)[:k]
        nearest_labels = [y_train[idx] for idx in nearest_neighbors]
        predicted_label = max(set(nearest_labels), key=nearest_labels.count)
        y_pred.append(predicted_label)
    return y_pred

# Path to the directory containing the image folders
train_dir = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample images"

# Load dataset
X, y = load_dataset(train_dir)

# Convert string labels to numerical values if needed
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply SMOTE to handle class imbalance if needed
sm = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Train KNN model
knn_model = knn(X_train_resampled, y_train_resampled, X_test)

# Evaluate the classifier
print('Accuracy {:.2f}'.format(np.mean(knn_model == y_test)))
conf_matrix = confusion_matrix(y_test, knn_model)
np.set_printoptions(threshold=np.inf)
print("Confusion Matrix:")
print(conf_matrix)
class_report = classification_report(y_test, knn_model)
print("Classification Report:\n", class_report)





# Load the trained classifier from the file
save_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\ledom.joblib'

try:
    loaded_clf = joblib.load(save_path)
    print("Classifier loaded successfully.")
except Exception as e:
    print("Error loading classifier:", e)
# Save the trained classifier to a file
joblib.dump(knn_model, save_path)
print("Classifier saved to:", save_path)

# Now classify a single input image
image_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample_image\7\im__21.png"
img = cv2.imread(image_path)
hog_features = extract_hog_features(img)
predicted_label = loaded_clf.predict([hog_features])
print("Predicted label:", predicted_label[0])
