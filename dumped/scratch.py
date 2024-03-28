from PIL import Image
import numpy as np

# Read the JPEG image
img_path = "C:\\Users\\amans\\OneDrive\\Desktop\\stat_project\\images\\alpha1.jpeg"
img = Image.open(img_path)

# Convert the image to grayscale
img_grey = img.convert('L')

# Convert the grayscale image to numpy array
img_array = np.array(img_grey)

# Calculate average grayscale value
avg = np.mean(img_array)

# Threshold for binarization
threshold = 0.9 * avg

# Binarization
binary_img = np.where(img_array >= threshold, 1, 0)

# Convert to 3-channel grayscale image
s_grey = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
for i in range(3):
    s_grey[:, :, i] = binary_img * 255  # Scale to 0-255 range

# Plot the grayscale image
Image.fromarray(s_grey).show()
