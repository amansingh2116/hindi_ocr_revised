import cv2
import numpy as np
import joblib
from pre import preprocess_image, display
from skimage.feature import hog
import os

def display(img, dpi=80):
  if img is None:
    print("Error: Could not read image from", img)
    return
  height, width = img.shape[:2] # Get image dimensions
  # Calculate aspect ratio preserving figure size in inches
  fig_width = width / float(dpi)
  fig_height = height / float(dpi)
  # Create a new window with calculated size
  cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create resizable window
  cv2.resizeWindow("Image", int(fig_width * 100), int(fig_height * 100))  # Resize in pixels
  cv2.imshow("Image", img) # Display the image on the created window
  cv2.waitKey(0) # Wait for a key press to close the window
  cv2.destroyAllWindows() # Close all windows


class CustomKNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for test_sample in X_test:
            distances = [np.linalg.norm(train_sample - test_sample) for train_sample in self.X_train]
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[idx] for idx in nearest_neighbors]
            predicted_label = max(set(nearest_labels), key=nearest_labels.count)
            y_pred.append(predicted_label)
        return y_pred




# Load the trained classifier from the joblib file
save_path = r'submit\model.joblib'
# Load the trained classifier from the file
try:
    loaded_clf, loaded_label_encoder = joblib.load(save_path)
    print("Classifier loaded successfully.")
except Exception as e:
    print("Error loading classifier:", e)

# Load default character images
default_characters_path = r"submit\final chars"



default_characters = {}
for filename in os.listdir(default_characters_path):
    label = int(filename.split('.')[0])  # Extract label from filename
    default_character_image = cv2.imread(os.path.join(default_characters_path, filename))
    default_characters[int(label)] = default_character_image


# Function to extract HOG features from an image
def extract_hog_features(image):
    # Preprocess the image
    img_res = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
    hog_img = hog(img_res, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    return hog_img

def predict_character(image):
    # Extract HOG features from the image
    hog_features = extract_hog_features(image)
    
    # Predict the label for the image using the loaded classifier
    predicted_label = loaded_clf.predict([hog_features])
    
    return loaded_label_encoder.inverse_transform(predicted_label)[0]

def ocr(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Calculate the resizing ratio based on both width and height
    max_display_width = 1000  # Maximum width for display
    max_display_height = 800  # Maximum height for display
    width_ratio = max_display_width / image.shape[1]
    height_ratio = max_display_height / image.shape[0]
    resizing_ratio = min(width_ratio, height_ratio)
    
    # Resize the image to fit the screen without cropping
    resized_image = cv2.resize(preprocessed_image, None, fx=resizing_ratio, fy=resizing_ratio, interpolation=cv2.INTER_AREA)
    
    # # Convert the image to grayscale
    # gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh_img = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Dilation to increase border width
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Expand the bounding box
        border_width = 10  # Adjust border width as needed
        x -= border_width
        y -= border_width
        w += 2 * border_width
        h += 2 * border_width
        
        # Ensure the coordinates are within the image boundaries
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, resized_image.shape[1] - x)
        h = min(h, resized_image.shape[0] - y)
        # Extract the character image
        character_image = resized_image[y:y+h, x:x+w]
        
        # Predict character
        predicted_label = predict_character(character_image)
        
        try:
            # Get the default character image based on the predicted label
            default_character = default_characters[int(predicted_label)]
        except KeyError:
            print("Character not found for label:", predicted_label)
            continue
        
        # Resize the default character image to match the size of the bounding box
        default_character_resized = cv2.resize(default_character, (w, h))
        
        # Convert default character image to grayscale
        default_character_resized_gray = cv2.cvtColor(default_character_resized, cv2.COLOR_BGR2GRAY)
        
        # Overlay the default character image onto the original image
        resized_image[y:y+h, x:x+w] = default_character_resized_gray
        
        # Draw bounding box
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (128, 0, 128), 2)  # Purple color
        
    # Display the result
    display(resized_image)


# Example usage
image_path = r"characters\final_char.jpeg"
ocr(image_path)