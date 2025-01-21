import cv2
import os
import numpy as np
from ultralytics import YOLO

# Callback function for trackbars (does nothing, needed for trackbar creation)
def nothing(x):
    pass

# Load the YOLOv9c model
model = YOLO('../model/runs/segment/train/weights/best.pt')

# Specify the test images directory
image_dir = "../model/datasets/test/images"

# Define a scaling factor of imshow() for displaying images
scale = 0.5  # 50% of the original size

# Check if the directory exists
if not os.path.exists(image_dir):
    print(f"Directory '{image_dir}' does not exist!")
    exit()

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

if not image_files:
    print("No image files found in the directory!")
    exit()

# Create a window for the trackbars
cv2.namedWindow("Trackbars")

# Initialize trackbars for lower and upper HSV bounds
cv2.createTrackbar("Lower H", "Trackbars", 33, 179, nothing)  # Hue range: 0-179
cv2.createTrackbar("Lower S", "Trackbars", 0, 255, nothing)   # Saturation range: 0-255
cv2.createTrackbar("Lower V", "Trackbars", 0, 255, nothing)   # Value range: 0-255
cv2.createTrackbar("Upper H", "Trackbars", 85, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

# Process and display each image
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Processing: {image_file}")
    
    # Load the original image
    image = cv2.imread(image_path)
    
    # Run inference on the image
    results = model(image_path)
    
    # Get the processed image with overlays (detections)
    detection_image = results[0].plot()  # Returns the image with bounding boxes and masks drawn
    
    # Convert the image from RGB to BGR for OpenCV display
    detection_image = cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR)
    
    # Configure each detection in the image
    if results[0].masks is not None:
        # Extract confidence scores
        masks = results[0].masks.data.cpu().numpy()  # Shape: (N, H, W), N masks
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

        # Find the index of the highest confidence
        highest_conf_index = np.argmax(scores)

        # Get the mask with the highest confidence
        mask = masks[highest_conf_index]  # Shape: (H, W)

        # Convert the mask to 3-channel (RGB) for visualization
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)

        # Create the segmented image (apply the mask to the original image)
        segmented_image = (image * mask_3d).astype(np.uint8)
        
        # Resize the images for display
        segmented_image_resized = cv2.resize(segmented_image, (0, 0), fx=scale, fy=scale)
        
        # Display the segmented region
        cv2.imshow("Segmented ROI (Masked)", segmented_image_resized)


        # Real-time HSV mask adjustments based on trackbar values
        while True:
            # Get trackbar positions for HSV bounds
            lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
            lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
            lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
            upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
            upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
            upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

            # Define the HSV bounds for greenish colors
            lower_green = np.array([lower_h, lower_s, lower_v])
            upper_green = np.array([upper_h, upper_s, upper_v])

            # Convert the mask to HSV
            hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

            # Remove all pitch white (255, 255, 255)
            non_white_and_black_mask = ~((segmented_image == [255, 255, 255]).all(axis=2) | (segmented_image == [0, 0, 0]).all(axis=2))

            # Apply the mask to remove white and black pixels
            filtered_image = segmented_image.copy()
            filtered_image[~non_white_and_black_mask] = [0, 0, 0]  # Set excluded pixels to black

            # Create a mask for greenish pixels
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            # Count greenish pixels
            green_pixel_count = np.count_nonzero(green_mask & non_white_and_black_mask)

            # Create an image showing only greenish pixels
            green_only_image = cv2.bitwise_and(segmented_image, segmented_image, mask=green_mask)

            # Create a mask for non-greenish pixels
            non_green_mask = cv2.bitwise_not(green_mask)

            # Count non-greenish pixels
            non_green_pixel_count = np.count_nonzero(non_green_mask & non_white_and_black_mask)

            # Create an image showing only non-greenish pixels
            non_green_only_image = cv2.bitwise_and(filtered_image, filtered_image, mask=non_green_mask)


            # Resize the images for display
            green_only_image_resized = cv2.resize(green_only_image, (0, 0), fx=scale, fy=scale)
            non_green_only_image_resized = cv2.resize(non_green_only_image, (0, 0), fx=scale, fy=scale)
            detection_image_resized = cv2.resize(detection_image, (0, 0), fx=scale, fy=scale)

            # Display the resized images
            cv2.imshow("Greenish Pixels", green_only_image_resized)
            cv2.imshow("Non-Green Pixels", non_green_only_image_resized)
            cv2.imshow("Detections", detection_image_resized)

            # Print the pixel counts
            print(f"Greenish Pixels: {green_pixel_count}")
            print(f"Non-Green Pixels: {non_green_pixel_count}")

            # Press 'r' to reset, 'n' for next image, or 'ESC' to exit
            key = cv2.waitKey(1)
            if key == 114:  # ASCII value of 'r'
                break  # Reset HSV adjustment loop
            elif key == 110:  # ASCII value of 'n' for next image
                break  # Proceed to the next image
            elif key == 27:  # ASCII value of 'ESC'
                exit()  # Exit the entire program


    else:
        print("No masks found in the image!")
 

    # Wait for a key press
    key = cv2.waitKey(0)
    if key == 27:  # Press 'ESC' to exit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
