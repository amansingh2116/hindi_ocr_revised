# code to implement preprocessing of image to achieve optical character recognition

# installing necessary libraries
import cv2
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt



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



# Resize the input image while preserving the aspect ratio.
def resize_image(image, scale_factor=None, width=None, height=None):

    if scale_factor is not None:
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    elif width is not None and height is not None:
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    elif width is not None:
        aspect_ratio = float(image.shape[1]) / float(image.shape[0])
        resized_image = cv2.resize(image, (width, int(width / aspect_ratio)), interpolation=cv2.INTER_AREA)
    elif height is not None:
        aspect_ratio = float(image.shape[1]) / float(image.shape[0])
        resized_image = cv2.resize(image, (int(height * aspect_ratio), height), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError("Please provide either scale_factor, width, or height.")

    return resized_image


# function to sharpen the input image.
def sharpen_image(image, strength=1):
    """
    Args:
    image: Input image (numpy array).
    strength: Strength of the sharpening effect (optional, default=1).
                Increase strength for stronger sharpening, decrease for weaker sharpening.
    """
    kernel = np.array([[0, -1, 0], [-1, 5 * strength, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image



# function to adjust the brightness and contrast of the input image.
def adjust_brightness_contrast(image, brightness=0, contrast=1):
    """
    Args:
    image: Input image (numpy array).
    brightness: Value to adjust the brightness (optional, default=0).
                  Positive values increase brightness, negative decrease.
    contrast: Value to adjust the contrast (optional, default=1).
                Values greater than 1 increase contrast, less than 1 decrease.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image



# function to invert images
def invert(file):
    invert_img = cv2.bitwise_not(file) 
    return invert_img


# function for thresholding image to remove noise and lines
def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# function to binarize image
def binarize(file):
    # Convert image to grayscale if it has multiple channels
    if len(file.shape) > 2:
        grey_img = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    else:
        grey_img = file  # Image is already grayscale
    
    # Apply binary threshold
    threshold, bin_img = cv2.threshold(grey_img, 150, 140, cv2.THRESH_BINARY)
    
    return bin_img



# function to remove noise
def noise_remove(file):
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(file, kernel, iterations=1)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    return (img)



# Dilation enlarges light objects in an image (regardless of whether you use positive or negative offset values). Dilation eliminates small dark gaps and holes and enlarges lighter features such as thin lines and fine detail.
# Erosion shrinks light objects in an image (regardless of whether you use positive or negative offset values). Erosion eliminates lighter features such as thin lines, fine detail, and small islands, and enlarges dark holes and gaps.
# In a morphological operation, the origin pixel of the structuring element is superimposed on a pixel in the source image.
# 1. For a dilation operation, the offset value at each pixel location in the structuring element is added to the value of its corresponding pixel in the source image. This yields a sum for each pixel location in the structuring element. The result is the maximum of these sums.
# 2.For an erosion operation, the offset value at each pixel location in the structuring element is subtracted from the value of its corresponding pixel in the source image. This yields a difference for each pixel location in the structuring element. The result is the minimum of these differences.


# function to erode image
def thin_font(file):
    img = invert(file)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = invert(img)
    return (img)

# function to dilate image
def thick_font(file):
    img = invert(file)
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=5)
    img = invert(img)
    return (img)



# function to deskew 
def deskew(file):
    canny = cv2.Canny(file,50,150) # edge detection

    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90,minLineLength=100,maxLineGap=10) # Hough Lines Transform
    if lines is None:
        print("No lines detected in the image.")
        return file
    drawing = np.zeros(file.shape[:], dtype=np.uint8)
    maxY=0
    degree_of_bottomline=0
    index=0
    for line in lines:        
        x1, y1, x2, y2 = line[0]            
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        k = float(y1-y2)/(x1-x2)
        degree = np.degrees(math.atan(k))
        if index==0:
            maxY=y1
            degree_of_bottomline=degree # take the degree of the line at the bottom
        else:        
            if y1>maxY:
                maxY=y1
                degree_of_bottomline=degree
        index=index+1

    img=Image.fromarray(file)
    rotateImg = img.rotate(degree_of_bottomline)
    rotateImg_cv = np.array(rotateImg)
    cv2.waitKey()
    return rotateImg_cv


# Function to remove borders
def remove_borders(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cntsSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y + h, x:x + w]
        return crop
    else:
        return img  # Return the original image if no contours are found


#function to add border
def add_border(file):
    color = [255, 255, 255] # 'cause purple!
    # border widths; I set them all to 150
    top, bottom, left, right = [150]*4
    return cv2.copyMakeBorder(file, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)




def preprocess_image(img):

    # resiz_img = resize_image(img, 1)
    # brightness = 1.1  # Adjust as desired
    # contrast = 1  # Adjust as desired
    adjusted_img = adjust_brightness_contrast(img)
    sharp_img = sharpen_image(adjusted_img)
    # inverted_img = invert(img)
    thres_img= thresholding(sharp_img)
    # binary_img = binarize(thres_img)
    noise_rem = noise_remove(thres_img)
    erode_img = thin_font(noise_rem)
    # dilate_img = thick_font(noise_rem)
    # deskew_img = deskew(dilate_img)
    no_border = remove_borders(erode_img)
    yes_border = add_border(no_border)

    return yes_border

# Plot the image
file_name = r"C:\Users\amans\OneDrive\Pictures\Screenshots\Screenshot 2024-05-12 081927.png"
original_image = cv2.imread(file_name)
# output= preprocess_image(original_image)
# image_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB
# plt.imshow(image_rgb)
# plt.axis('off')  # Turn off axis
# plt.show()

import cv2
import matplotlib.pyplot as plt

def character_segmentation(image):
    # preprocess image
    image = preprocess_image(image)

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

    # Process each contour (character) and save as separate image
    segmented_images = []
    for i, cnt in enumerate(contours):
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

        # Draw bounding box on the original image
        cv2.rectangle(image, (int(x / resizing_ratio), int(y / resizing_ratio)), 
                      (int((x + w) / resizing_ratio), int((y + h) / resizing_ratio)), 
                      (0, 255, 0), 2)

    return image

# Perform character segmentation and overlay bounding boxes
output_image = character_segmentation(original_image)

# Convert images from BGR to RGB for displaying with Matplotlib
input_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Plot both images side by side
plt.figure(figsize=(12, 6))

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(input_image_rgb)
plt.title('Original Image')
plt.axis('off')

# Plot image with bounding boxes
plt.subplot(1, 2, 2)
plt.imshow(output_image_rgb)
plt.title('Segmented Image with Bounding Boxes')
plt.axis('off')

plt.show()
