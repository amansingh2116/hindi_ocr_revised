# Rotate images in a folder

import os
import cv2

def rotate_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Get the height and width of the image
    height, width = image.shape[:2]
    
    # Compute the center of the image
    center = (width / 2, height / 2)
    
    # Define the rotation angle (180 degrees)
    angle = 180
    
    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

def rotate_images_in_folder(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, filename)
            
            # Rotate the image
            rotated_image = rotate_image(image_path)
            
            # Save the rotated image, overwriting the original
            cv2.imwrite(image_path, rotated_image)
            print(f"Rotated and saved: {filename}")

# Path to the folder containing images
folder_path = "input_folder"

# Rotate all images in the folder
rotate_images_in_folder(folder_path)
