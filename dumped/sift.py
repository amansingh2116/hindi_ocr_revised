import cv2
import matplotlib.pyplot as plt

# Load the train image and display it
train_image = cv2.imread("C:\\Users\\amans\\OneDrive\\Documents\\GitHub\\stat_sem2_project\\images\\alpha1.jpeg")
# Convert the color to be inline with matplotlib. OpenCV uses the BGR sequence, whereas Matplotlib interprets as RGB
train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)


# Load the test image and display it
test_image = cv2.imread("C:\\Users\\amans\\OneDrive\\Documents\\GitHub\\stat_sem2_project\\images\\saurabh1bet.jpg")
# Convert the color to be inline with matplotlib. OpenCV uses the BGR sequence, whereas Matplotlib interprets as RGB
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)


# Create ORB (Oriented FAST and Rotated BRIEF) using the SIFT
orb = cv2.xfeatures2d.SIFT_create()
# Find the keypoints and descriptors of both train and test images
kp1, des1 = orb.detectAndCompute(train_image, None)
kp2, des2 = orb.detectAndCompute(test_image, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
# If the distance between the matches is less than 75% then it is considered as a good match, else discard
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# In the below line, we are considering the top 100 matches and hence good[:100]
img3 = cv2.drawMatchesKnn(train_image, kp1, test_image, kp2, good[:100], None, flags=2)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.imshow(img3)
plt.show()
