import os
import cv2

def display_images_in_folder(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Display the image
        cv2.imshow("Image", image)
        
        # Wait for the user's input
        key = cv2.waitKey(0)
        
        # Check if the spacebar or 'd' key is pressed
        if key == ord('d'):
            # Delete the image file
            os.remove(image_path)
            print(f"Deleted: {image_file}")
        elif key == 32:  # Spacebar key
            continue
        
        # Close the window
        cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample_image\up tri"

# Display images in the folder one by one
display_images_in_folder(folder_path)
