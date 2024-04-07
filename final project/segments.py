import cv2
import numpy as np

def character_segmentation(image):
    # Calculate the resizing ratio based on both width and height
    max_display_width = 1000  # Maximum width for display
    max_display_height = 800  # Maximum height for display
    width_ratio = max_display_width / image.shape[1]
    height_ratio = max_display_height / image.shape[0]
    resizing_ratio = min(width_ratio, height_ratio)
    
    # Resize the image to fit the screen without cropping
    resized_image = cv2.resize(image, None, fx=resizing_ratio, fy=resizing_ratio, interpolation=cv2.INTER_AREA)
    
    # Apply thresholding
    _, thresh_img = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Dilation to increase border width
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract segmented character images along with bounding boxes
    segmented_characters = []
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
        
        # Extract the segmented character image
        segmented_image = resized_image[y:y+h, x:x+w]

        # Append the segmented character image along with its bounding box coordinates to the list
        segmented_characters.append([segmented_image, x, y, w, h])
    
    return segmented_characters

# Example usage
# image_path = "C:\\Users\\amans\\OneDrive\\Documents\\GitHub\\stat_sem2_project\\images\\saurabh2bet.jpg"
# original_image = cv2.imread(image_path)
# segmented_characters = character_segmentation(original_image)

# Display each segmented character image one by one
# for character_img in segmented_characters:
#     cv2.imshow('hello', character_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()