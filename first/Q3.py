import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread('trees.jpeg')

# Step 2: Convert image from BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Display the original image
plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Step 4: Separate the color channels
R, G, B = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]

# Step 5: Display each color channel
plt.subplot(2, 2, 2)
plt.imshow(R, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(G, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(B, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 6: Compute histograms
hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])
hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])
hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])
hist_total = cv2.calcHist([image_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# Step 7: Plot histograms
plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
plt.plot(hist_R, color='r')
plt.title('Red Channel Histogram')
plt.xlim([0, 256])

plt.subplot(2, 2, 2)
plt.plot(hist_G, color='g')
plt.title('Green Channel Histogram')
plt.xlim([0, 256])

plt.subplot(2, 2, 3)
plt.plot(hist_B, color='b')
plt.title('Blue Channel Histogram')
plt.xlim([0, 256])

plt.subplot(2, 2, 4)
plt.hist(image_rgb.ravel(), bins=256, color='black', alpha=0.5, label='Overall')
plt.title('Overall Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()


# Step 1: Read the image
image = cv2.imread('abraham.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Display the original image
plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Step 3: Draw histogram of the original image
hist_orig = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.subplot(2, 2, 2)
plt.plot(hist_orig, color='black')
plt.title('Histogram of Original Image')
plt.xlim([0, 256])

# Step 4: Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)


# Step 5: Display the equalized image
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Step 6: Draw histogram of the equalized image
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.subplot(2, 2, 4)
plt.plot(hist_equalized, color='black')
plt.title('Histogram of Equalized Image')
plt.xlim([0, 256])

original_cumulative_histogram = np.cumsum(hist_orig)
original_cumulative_histogram = original_cumulative_histogram / original_cumulative_histogram.max() * 255

equalized_cumulative_histogram = np.cumsum(hist_equalized)
equalized_cumulative_histogram = equalized_cumulative_histogram / equalized_cumulative_histogram.max() * 255

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))

plt.subplot(1, 2, 1)
plt.title("Cumulative Grayscale Histogram of Original")
plt.xlabel("Intensity Value")
plt.ylabel("Cumulative Frequency")
plt.plot(original_cumulative_histogram, color='r')

plt.subplot(1, 2, 2)
plt.title("Cumulative Grayscale Histogram of Equalized")
plt.xlabel("Intensity Value")
plt.ylabel("Cumulative Frequency")
plt.plot(equalized_cumulative_histogram, color='g')

plt.tight_layout()
plt.show()
