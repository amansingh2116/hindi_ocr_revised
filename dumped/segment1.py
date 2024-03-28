import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('C://Users//amans//OneDrive//Desktop//stat_project//images//alpha1.jpeg')

# Convert the image to grayscale
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive Thresholding (improves handling of uneven lighting)
thresh_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Dilation (adjust kernel size as needed)
kernel = np.ones((3, 5), np.uint8)  # Smaller kernel for more precise word separation
dilated = cv2.dilate(thresh_img, kernel, iterations=1)

# Erosion (optional, reduces noise)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Line Segmentation
(contours, _) = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])


# Word Segmentation
img3 = img.copy()
words_list = []

for line in sorted_contours_lines:
    # Region of Interest (ROI) for each line
    x, y, w, h = cv2.boundingRect(line)
    roi_line = eroded[y:y+h, x:x+w]
    
    # Find contours for each line
    (word_contours, _) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for word_contour in word_contours:
        # Get bounding box for the word contour
        word_x, word_y, word_w, word_h = cv2.boundingRect(word_contour)
    
        # Filter out small contours (adjust threshold based on font size)
        if cv2.contourArea(word_contour) < 50:  # Experiment with this value
            continue
    
        # Improve bounding box accuracy (optional)
        # You can explore minimum enclosing rectangle or convex hull here

        # Extract the word region from the original image
        word_roi = img[y + word_y:y + word_y + word_h, x + word_x:x + word_x + word_w]
    
        # Draw rectangles around words
        cv2.rectangle(img3, (x + word_x, y + word_y), (x + word_x + word_w, y + word_y + word_h), (255, 255, 100), 2)
    
        # Append word region to the list
        words_list.append(word_roi)

# Display the result with lines and words
plt.imshow(img3)
plt.show()
