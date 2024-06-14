import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def T1(image):
    L = 256  # Assuming 8-bit grayscale image
    
    # Define the transformation function
    def transform_pixel(pixel):
        return -1*pixel + (L-1)

    # Vectorize the transformation function
    vectorized_transform = np.vectorize(transform_pixel)
    
    # Apply the transformation
    stretched_image = vectorized_transform(image)
    
    # Clip values to ensure they are within [0, 255] and convert back to uint8
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    
    return stretched_image


def T2(image, r1, r2):
    L = 255  # Assuming 8-bit grayscale image
    
    # Define the transformation function
    def transform_pixel(pixel):
        if pixel <= r1*L:
            return pixel
        elif pixel <= r2*L:
            return 0
        else:
            return pixel

    # Vectorize the transformation function
    vectorized_transform = np.vectorize(transform_pixel)
    
    # Apply the transformation
    stretched_image = vectorized_transform(image)
    
    # Clip values to ensure they are within [0, 255] and convert back to uint8
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    
    return stretched_image


def T3(image, r1, r2):
    L = 255  # Assuming 8-bit grayscale image
    
    # Define the transformation function
    def transform_pixel(pixel):
        if pixel <= r1*L:
            return 0
        elif pixel <= r2*L:
            return pixel
        else:
            return L

    # Vectorize the transformation function
    vectorized_transform = np.vectorize(transform_pixel)
    
    # Apply the transformation
    stretched_image = vectorized_transform(image)
    
    # Clip values to ensure they are within [0, 255] and convert back to uint8
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    
    return stretched_image


def T4(image, r1, r2, s1):
    L = 255  # Assuming 8-bit grayscale image
    
    # Define the transformation function
    def transform_pixel(pixel):
        if pixel <= r1*L:
            return (s1/r1)*pixel
        elif pixel <= r2*L:
            return pixel
        else:
            return ((1 - s1)/(1- r2))*(pixel-r2) + s1

    # Vectorize the transformation function
    vectorized_transform = np.vectorize(transform_pixel)
    
    # Apply the transformation
    stretched_image = vectorized_transform(image)
    
    # Clip values to ensure they are within [0, 255] and convert back to uint8
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    
    return stretched_image


def T5(image):   
    # Normalize the pixel values to the range [0, 1]
    normalized_image = image / 255.0
    
    # Apply the "2th power" transformation (square the pixel values)
    transformed_image = np.power(normalized_image, 2)
    
    # Scale the transformed image back to the range [0, 255]
    scaled_transformed_image = np.uint8(transformed_image * 255)

    return scaled_transformed_image


def T6(image):   
    # Normalize the pixel values to the range [0, 1]
    normalized_image = image / 255.0
    
    # Apply the "2th root" transformation (square root of the pixel values)
    transformed_image = np.sqrt(normalized_image)
    
    # Scale the transformed image back to the range [0, 255]
    scaled_transformed_image = np.uint8(transformed_image * 255)

    return scaled_transformed_image


image = cv2.imread('image1.jfif', cv2.IMREAD_GRAYSCALE)

images = {
    "Original": image,
    "T1": T1(image),
    "T2": T2(image, 0.2, 0.55),
    "T3": T3(image, 0.4, 0.55),
    "T4": T4(image, 0.2, 0.55, 0.3),
    "T5": T5(image),
    "T6": T6(image)
}

# Plot all images in one window with the original image on top
plt.figure(figsize=(12, 8))

# Plot the original image
ax = plt.subplot(3, 3, 2)
plt.title("Original")
plt.imshow(images["Original"], cmap='gray')
plt.axis('off')
# Draw a box around the original image
rect = patches.Rectangle((0, 0), images["Original"].shape[1], images["Original"].shape[0], linewidth=3, edgecolor='red', facecolor='none')
ax.add_patch(rect)

# Plot the transformed images
for i, (title, img) in enumerate(images.items()):
    if title != "Original":
        ax = plt.subplot(3, 3, i + 3)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        # Draw a box around each transformed image
        rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.tight_layout()
plt.show()