import cv2
import numpy as np
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt

# Load the classifier
clf = joblib.load(r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\OCR\classifi.joblib")

# Function to perform character segmentation and classify characters
def segment_and_classify(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return
    
    # Calculate the resizing ratio based on both width and height
    max_display_width = 1000  # Maximum width for display
    max_display_height = 800  # Maximum height for display
    width_ratio = max_display_width / image.shape[1]
    height_ratio = max_display_height / image.shape[0]
    resizing_ratio = min(width_ratio, height_ratio)
    
    # Resize the image to fit the screen without cropping
    resized_image = cv2.resize(image, None, fx=resizing_ratio, fy=resizing_ratio, interpolation=cv2.INTER_AREA)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Dilation to increase border width
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store predicted characters row-wise
    row_chars = []
    prev_y = None
    
    # Draw bounding boxes with adjusted border width and classify characters
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
        
        # Extract character region
        char_region = resized_image[y:y+h, x:x+w]
        
        # Convert the character region to grayscale
        char_gray = cv2.cvtColor(char_region, cv2.COLOR_BGR2GRAY)
        
        # Resize the character region to match the size used during training
        char_resized = cv2.resize(char_gray, (64, 128))
        
        # Calculate HOG features
        hog_features = hog(char_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        
        # Predict character label
        predicted_label = clf.predict([hog_features])[0]
        
        # Store the predicted character
        if prev_y is not None and y != prev_y:
            print(''.join(row_chars))
            row_chars = []
        
        row_chars.append(predicted_label)
        prev_y = y
        
        # Draw the expanded bounding box
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        # Write the predicted label below the bounding box
        cv2.putText(resized_image, str(predicted_label), (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 250), 2)
    
    # Print the last row of characters
    print(''.join(row_chars))
    
    # Display the result
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis
    plt.show()

# Example usage
image_path = "C:\\Users\\amans\\OneDrive\\Documents\\GitHub\\stat_sem2_project\\images\\alpha1.jpeg"
segment_and_classify(image_path)