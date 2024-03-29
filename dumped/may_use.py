from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np

# Assume X_images contains your image data and y_labels contains corresponding labels
X_images = ''
y_labels = ''

# Reshape images if necessary and flatten them into feature vectors
X_features = np.reshape(X_images, (len(X_images), -1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Apply oversampling to balance the training set
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Now X_resampled and y_resampled contain the balanced training data
# Proceed with training your CNN model using the balanced data










import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

# Example image augmentation function
def augment_image(image):
    # Convert image to numpy array
    image_array = np.array(image)

    # Define augmentation pipeline
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Gaussian blur
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Gaussian noise
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # brightness and contrast changes
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scaling
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translation
            rotate=(-45, 45),  # rotation
            shear=(-16, 16),  # shear
            mode='edge'  # mode used for out of image pixels
        )
    ], random_order=True)  # apply augmentations in random order

    # Augment image
    augmented_image_array = seq(image=image_array)

    # Convert augmented numpy array back to PIL image
    augmented_image = Image.fromarray(augmented_image_array)

    return augmented_image

# Example usage
# Load an example image
input_image = Image.open("example_image.jpg")

# Augment the image
augmented_image = augment_image(input_image)

# Display the original and augmented images
input_image.show()
augmented_image.show()
