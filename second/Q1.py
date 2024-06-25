import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = 'Shapes.jpg'
image = cv2.imread(image_path)

# Convert the image to RGB (from BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to label detected shapes
def label_shapes(image):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    # Convert image to grayscale
    gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    # Apply threshold to get binary image

    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get the moments to calculate the center of the contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        # Determine the shape
        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            # Distinguish between square and rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif len(approx) == 5:
            shape_name = "Pentagon"
        elif len(approx) == 6:
            shape_name = "Hexagon"
        else:
            shape_name = "Circle"
        
        
        # Draw the contour and the name of the shape
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, shape_name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


# Function to remove non-square shapes
def remove_non_squares(image):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    # Convert image to grayscale
    gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    # Apply threshold to get binary image

    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    
    for contour in contours:
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    
    # Create the final image where only squares are retained
    result = cv2.bitwise_and(image, image, mask=mask)
    return result



#---------------------------------------------------------------------------------------
# Apply the function to label shapes
labeled_image = label_shapes(image.copy())

# Convert the image to RGB for display
labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)

#---------------------------------------------------------------------------------------
# Apply the function to remove non-squares
squares_image = remove_non_squares(image.copy())

# Convert the image to RGB for display
squares_image_rgb = cv2.cvtColor(squares_image, cv2.COLOR_BGR2RGB)


# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
plt.axis('off')

# Display the labeled image
plt.figure(figsize=(15, 8))
plt.subplot(1,2,1)
plt.imshow(labeled_image_rgb)

plt.subplot(1,2,2)
plt.imshow(squares_image_rgb)
plt.axis('off')
plt.show()

