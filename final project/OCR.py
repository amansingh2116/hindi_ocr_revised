# logistic regression
import cv2
import joblib
from skimage.feature import hog 

# Load the trained classifier from the joblib file
model_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\final project\model.joblib'
loaded_model = joblib.load(model_path)

print("Type of loaded_model:", type(loaded_model))  # Add this line to check the type

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Preprocess the image
    img_res = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    hog_img = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    return hog_img

def predict_character(image_path):

    # Load a new image

    img = cv2.imread(image_path)

    # Extract HOG features from the new image
    hog_features = extract_hog_features(img)

    # Predict the label for the new image using the loaded classifier
    predicted_label = loaded_model.predict([hog_features])

    print("Predicted label:", predicted_label[0])



image_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample images\5\im__4.png"
predict_character(image_path)