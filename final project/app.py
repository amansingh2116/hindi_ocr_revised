import cv2
import numpy as np
import joblib
from preprocessing import preprocess_image
from skimage.feature import hog

# Load the trained classifier from the joblib file
model_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\final project\model.joblib'
loaded_model = joblib.load(model_path)

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
    predicted_label = loaded_model.predict([hog_features])
    
    return str(predicted_label[0])

def character_segmentation(image_path):
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
    
    
    # Apply thresholding
    _, thresh_img = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Dilation to increase border width
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes with adjusted border width and predict characters
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # # Expand the bounding box
        # border_width = 10  # Adjust border width as needed
        # x -= border_width
        # y -= border_width
        # w += 2 * border_width
        # h += 2 * border_width
        
        # # Ensure the coordinates are within the image boundaries
        # x = max(x, 0)
        # y = max(y, 0)
        # w = min(w, resized_image.shape[1] - x)
        # h = min(h, resized_image.shape[0] - y)
        
        # Draw the expanded bounding box
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (155, 0, 255), 2)
        
        # Predict character and draw text below the bounding box
        predicted_char = predict_character(resized_image[y:y+h, x:x+w])
        cv2.putText(resized_image, predicted_char, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Display the result
    cv2.imshow('Character Segmentation', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\images\alpha1.jpeg"
character_segmentation(image_path)
