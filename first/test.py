import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("abraham.jpg", cv2.IMREAD_GRAYSCALE)
# h, w = img.shape
# cv2.imshow("org", img)

# kernel = np.array([[1,0,0],
#                    [0.3,1,0]
#                    ], dtype=np.float32)

# t = 90
# # Calculate the center of the image
# center = (w // 2, h // 2)

# # Get the rotation matrix using cv2.getRotationMatrix2D
# rotation_matrix = cv2.getRotationMatrix2D(center, t, 1)

# # Apply the affine transformation
# trans = cv2.warpAffine(img, kernel, (w, h))

# # Display the transformed image
# cv2.imshow("Transformed", trans)

# # Wait for a key press and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load the image in grayscale
# img = cv2.imread('hand.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
# Parameters: source image, kernel size (must be odd and positive), standard deviation in X direction
# Load the colored image


# Apply Gaussian blur
# Parameters: source image, kernel size, standard deviation in X direction
# gaussian_blur = cv2.GaussianBlur(img, (15, 15), 0)

# # Display the original and blurred images
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Gaussian Blurred Image')
# plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# Step 2: Apply Gaussian filter for noise reduction
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 5)

# Step 3: Apply Sobel operator to get gradients in X and Y directions
Gx = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)

# Step 4: Calculate the gradient magnitude
magnitude = cv2.magnitude(Gx, Gy)

# Step 5: Calculate the gradient direction
direction = cv2.phase(Gx, Gy, angleInDegrees=True)

# Normalize the magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = np.uint8(magnitude)

# Normalize the direction for display
direction = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX)
direction = np.uint8(direction)

# Display the results
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Gaussian Blurred Image')
plt.imshow(gaussian_blur, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Gradient in X direction')
plt.imshow(np.absolute(Gx), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Gradient in Y direction')
plt.imshow(np.absolute(Gy), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Gradient Magnitude')
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Gradient Direction')
plt.imshow(direction, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()