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

# Read the image
im = cv2.imread("C:\\Users\\amans\\OneDrive\\Desktop\\stuff\\output2.jpeg")
im = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

# Convert the image to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Apply dilation
kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

# Draw bounding boxes directly on the original image
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Draw bounding box with increased width
    cv2.rectangle(im, (x, y), (x + w, y + h), (90, 0, 255), 3)  # Increase thickness parameter to increase width

# Display the image with bounding boxes
display(im)