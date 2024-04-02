import cv2
import numpy as np
import os
import csv
import shutil

def character_segmentation(image_path, character_prefix):
    # Load the image
    image = cv2.imread(image_path)

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

        # Crop the character from the image
        character_image = resized_image[y:y+h, x:x+w]

        # Generate the filename with prefix and incremental number
        output_filename = f"{character_prefix}_{i+1}.png"
        segmented_images.append((output_filename, character_image))

    return segmented_images

def process_images(file_path, output_folder, character_prefix):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read file paths from the input file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_path = row[0]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_folder = os.path.join(output_folder, image_name)
            if not os.path.exists(image_output_folder):
                os.makedirs(image_output_folder)

            output_images = character_segmentation(image_path, character_prefix)

            # Move segmented character images to output folder
            for output_filename, character_image in output_images:
                output_path = os.path.join(image_output_folder, output_filename)
                cv2.imwrite(output_path, character_image)

# Example usage
input_file_path = "TOOLS\\path.csv"  # Path to the input file containing image paths
output_folder_path = "C:\\Users\\amans\\OneDrive\\Documents\\GitHub\\stat_sem2_project\\sample_image"  # Path to the folder where output images will be saved
character_prefix = "p"  # Prefix to be added to each segmented character filename
process_images(input_file_path, output_folder_path, character_prefix)
# delete all the wrong bounded boxes and keep only the characters in the folders