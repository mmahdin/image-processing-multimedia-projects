import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_biggest_contour_corners(image):
    
    # Convert to grayscale
    gray = image
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours are found
    if not contours:
        print("No contours found")
        return
    
    # Find the biggest contour by area
    biggest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon and get the corner points
    epsilon = 0.02 * cv2.arcLength(biggest_contour, True)
    approx = cv2.approxPolyDP(biggest_contour, epsilon, True)
    
    # Check if the approximation is a quadrilateral
    if len(approx) != 4:
        print("The biggest contour is not a quadrilateral")
        return
    
    # Print the corner points
    corners = approx.reshape(4, 2)
    print("The corners of the biggest contour are:")
    for corner in corners:
        print(tuple(corner))
    
    # # Optionally, draw the contour and corners on the image for visualization
    # cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    # for corner in corners:
    #     cv2.circle(image, tuple(corner), 5, (0, 0, 255), -1)
    
    # # Display the image
    # cv2.imshow("Biggest Contour with Corners", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def bitwise_or_images(image1, image2):
    if image1.shape != image2.shape:
        print("Error: The dimensions of the images do not match")
        return
    # Perform bitwise OR operation
    result = cv2.bitwise_and(image1, cv2.bitwise_not(image2))
    return result

def crop_largest_contour(image):
    # Convert to grayscale
    gray_image = image

    # Find contours
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def apply_reflection(image):
    reflected_image = cv2.flip(image, 1) 
    return reflected_image

def scale_image_horizontal(image, s):

    # Get original dimensions
    h, w = image.shape

    h, w = int(h*s[0]), int(w*s[1])

    # Resize the image
    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized_image

def apply_affine_transformation(image, angle, scale, tx, ty):
    # Get the image dimensions
    rows, cols = image.shape
    
    # Compute the center of the image
    center = (cols / 2, rows / 2)
    
    # Compute the transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply the translation to the transformation matrix
    rotation_matrix[0, 2] += tx  # adding translation in x direction
    rotation_matrix[1, 2] += ty  # adding translation in y direction
    
    # Apply the afficv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)ne transformation using the transformation matrix
    transformed_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    
    return transformed_image

def apply_shear_transformation(image, shx=0, shy=0):
    rows, cols = image.shape
    
    # Define the shear transformation matrix
    shear_matrix = np.array([[1, shx, 0],
                             [shy, 1, -200]], dtype=np.float32)
    
    # Apply the shear transformation using the transformation matrix
    transformed_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    return transformed_image

def move_image(image, tx, ty, cols, rows):
    move_matrix = np.array([[1, 0, tx],
                            [0, 1, ty]], dtype=np.float32)
    transformed_image = cv2.warpAffine(image, move_matrix, (cols, rows))
    return transformed_image



def main():
    # Load the original and transformed images
    img1 = cv2.imread('./Original_image.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./transformed_image.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    # T1
    reflected_image = apply_reflection(img1)

    # T2
    sheared_image = apply_shear_transformation(reflected_image, shx=0, shy=0.6)

    # T3
    resized_h_image = scale_image_horizontal(sheared_image, (0.93, 1.79))

    # T4
    affine_image = apply_affine_transformation(resized_h_image, angle = -11.5, scale = 0.2, tx=0, ty=0)

    # T5
    moved_image = move_image(affine_image, tx=-73.5   , ty=375, cols=w, rows=h)


    compare1 = bitwise_or_images(img2, moved_image)
    compare2 = bitwise_or_images(moved_image, img2)


    # Display the results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(reflected_image, cmap='gray')
    plt.title('Reflected image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(sheared_image, cmap='gray')
    plt.title('Sheared image')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(resized_h_image, cmap='gray')
    plt.title('Resized h image')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(affine_image, cmap='gray')
    plt.title('affine image')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(moved_image, cmap='gray')
    plt.title('Moved image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    
    # Create a figure with 2 rows and 3 columns
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(moved_image, cmap='gray')
    plt.title('Result')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Test')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(compare1, cmap='gray')
    plt.title('Test & (~Result)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(compare2, cmap='gray')
    plt.title('Result & (~Test)')
    plt.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
