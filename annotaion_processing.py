import cv2
import numpy as np


def fill_prediction_areas(image, predictions, color=(0, 255, 0)):
    for prediction in predictions:
        x_min, y_min, x_max, y_max = prediction
        # Define the points of the rectangle
        points = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        points = points.reshape((-1, 1, 2))

        # Draw a filled polygon on the image
        cv2.fillPoly(image, [points], color)

    return image


def fill_in_between_prediction_areas(image, predictions, color=(0, 255, 0)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV (narrow range around green)
    lower_green = np.array([55, 200, 200])  # Lower bound for green
    upper_green = np.array([65, 255, 255])  # Upper bound for green

    # Create a mask for the specified green color range
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the areas of the contours
    areas = [cv2.contourArea(contour) for contour in contours]

    # Print the range of areas
    if areas:
        min_area = min(areas)
        max_area = max(areas)
        print(f"The range of areas between the specified green color pixels is: {min_area} to {max_area}")
    else:
        print("No areas found between the specified green color pixels.")

    # Optionally, draw the contours on the original image for visualization
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
