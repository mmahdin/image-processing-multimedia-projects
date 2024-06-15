import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread('AerialView.jpeg', cv2.IMREAD_GRAYSCALE)

# Step 2: Apply Gaussian Filter
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Step 3: Apply Median Filter
median_blur = cv2.medianBlur(image, 5)

# Step 4: Apply Sharpening Filter
sharpening_kernel = np.array([[-1, -1, -1], 
                              [-1, 9, -1],
                              [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, sharpening_kernel)

# Step 5: Apply Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Step 6: Apply Canny Edge Detection
canny_edges = cv2.Canny(image, 100, 200)

# Step 7: Apply Sobel and Canny on Gaussian Blurred Image
sobel_gaussian = cv2.magnitude(cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3))
canny_gaussian = cv2.Canny(gaussian_blur, 100, 200)

# Step 8: Apply Sobel and Canny on Median Blurred Image
sobel_median = cv2.magnitude(cv2.Sobel(median_blur, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(median_blur, cv2.CV_64F, 0, 1, ksize=3))
canny_median = cv2.Canny(median_blur, 100, 200)

# Step 9: Apply Sobel and Canny on Sharpened Image
sobel_sharpened = cv2.magnitude(cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3))
canny_sharpened = cv2.Canny(sharpened, 100, 200)

# Display the results
plt.figure(figsize=(20, 15))

plt.subplot(3, 4, 1)
plt.imshow(gaussian_blur, cmap='gray')
plt.title('Gaussian Blurred')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(median_blur, cmap='gray')
plt.title('Median Blurred')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(sobel_gaussian, cmap='gray')
plt.title('Sobel on Gaussian')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(canny_gaussian, cmap='gray')
plt.title('Canny on Gaussian')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(sobel_median, cmap='gray')
plt.title('Sobel on Median')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(canny_median, cmap='gray')
plt.title('Canny on Median')
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(sobel_sharpened, cmap='gray')
plt.title('Sobel on Sharpened')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(canny_sharpened, cmap='gray')
plt.title('Canny on Sharpened')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel on Original')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny on Original')
plt.axis('off')

plt.tight_layout()
plt.show()
