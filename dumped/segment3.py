import cv2
import numpy as np

# function to display images
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


def character_segmentation(image):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Apply dilation
    horizontal_kernel = np.ones((1, 50), np.uint8)
    dilation = cv2.dilate(thresh, horizontal_kernel, iterations=1)
    
    # Find contours of characters
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Draw bounding rectangles around characters
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Increase width of rectangle sides
        rect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Display the image with bounding boxes
    # Display the image with bounding boxes
    display(image)

# Load the image
my_image = cv2.imread("C:\\Users\\amans\\OneDrive\\Desktop\\stuff\\output2.jpeg")

# Perform character segmentation
character_segmentation(my_image)
