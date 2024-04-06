import os
import cv2

def display_images_in_folder(folder_path):
    # Get a list of all image files in the folder and sort them alphabetically
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

    # Keep track of the current index
    current_index = 0

    while current_index < len(image_files):
        # Construct the full path to the image file
        image_file = image_files[current_index]
        image_path = os.path.join(folder_path, image_file)

        # Read the image
        image = cv2.imread(image_path)

        # Display the image
        cv2.imshow("Image", image)

        # Wait for the user's input
        key = cv2.waitKey(0)

        # Check if the spacebar, 'd', or 'b' key is pressed
        if key == ord('d'):
            # Delete the image file
            os.remove(image_path)
            print(f"Deleted: {image_file}")
            # Remove the deleted image from the list
            del image_files[current_index]
        elif key == ord('b'):
            # Move to the previous image if 'b' is pressed
            current_index = max(0, current_index - 1)
            continue
        elif key == 32:  # Spacebar key
            current_index += 1
            continue

        # Close the window
        cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = r"C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\sample_image\1"

# Display images in the folder one by one
display_images_in_folder(folder_path)
