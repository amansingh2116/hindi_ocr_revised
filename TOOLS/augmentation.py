# NOTE: this code will create augmented images of the dataset in the same folder whose path give to take dataset from.
import numpy as np
import cv2
import os

# Define the augmentation functions with limits on the transformations

def random_shift(image, w, h):
    max_shift_x = 0.02 * w  # Limit the shift to 2% of the width
    max_shift_y = 0.02 * h  # Limit the shift to 2% of the height
    shift_x = max_shift_x * np.random.uniform(-1, 1)
    shift_y = max_shift_y * np.random.uniform(-1, 1)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image


def curved_distortion(image, curve_type='rainbow'):
    h, w = image.shape
    new_image = np.zeros_like(image)
    max_y_shift = 0.05 * h  # Limit the vertical shift to 5% of the height
    for x in range(w):
        if curve_type == 'rainbow':
            y_shift = int(max_y_shift * np.sin(2 * np.pi * x / w))
        elif curve_type == 'inverted':
            y_shift = int(-max_y_shift * np.sin(2 * np.pi * x / w))
        for y in range(h):
            new_y = y + y_shift
            if 0 <= new_y < h:
                new_image[new_y, x] = image[y, x]
    
    return new_image

def sinusoidal_distortion(image):
    h, w = image.shape
    new_image = np.zeros_like(image)
    max_y_shift = 0.05 * h  # Limit the vertical shift to 5% of the height
    for x in range(w):
        y_shift = int(max_y_shift * np.sin(2 * np.pi * x / w))
        for y in range(h):
            new_y = y + y_shift
            if 0 <= new_y < h:
                new_image[new_y, x] = image[y, x]
    
    return new_image


# Function to augment images and save in the same folder
def augment_and_save_images_in_place(folder_path):
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                h, w = image.shape

                # Apply each augmentation and save in the same folder
                augmentations = [
                    ('shifted', random_shift(image, w, h)),
                    ('rainbow', curved_distortion(image, 'rainbow')),
                    ('inverted', curved_distortion(image, 'inverted')),
                    ('sinusoidal', sinusoidal_distortion(image))
                ]

                for aug_name, aug_image in augmentations:
                    new_file_name = os.path.join(subdir, f'{os.path.splitext(file)[0]}_{aug_name}.png')
                    cv2.imwrite(new_file_name, aug_image)

# Example usage
root_folder_path = r'sample images'
augment_and_save_images_in_place(root_folder_path)
