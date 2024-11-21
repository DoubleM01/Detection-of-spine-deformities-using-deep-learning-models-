import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow
import supervision as sv
import cv2
import Image_Converters
import annotaion_processing
import numpy as np
import matplotlib

vertebraeLocalization_initialized = False
vertebraeLocalization_modelID = "lumbar-st35n/2"
vertebraeLocalization_projectName = "lumbar-st35n"
#vertebraeLocalization_modelID = "segmentation-95qui/3"
#input_imageName = r"test/T1_0002_S8_png.rf.a407fe02a4f69f362148b65e8018122c.jpg"
input_imageName = "test.png"
Image_Converters.convert_png_2jpg(input_imageName)
input_imageName = input_imageName.replace(".png", ".jpg")
vertebraeLocalized_image = None
vertebraeLocalized_imageName = "VertebraeLocalized1.jpg"
#CLIENT = None
model = None
vertebraeLocations = []


def vertebrae_localization_preprocess(img_name):
    if not is_valid_image_type(img_name):
        #not finished 11:32 PM 13, June 2024
        ...


def is_valid_image_type(img_name):
    if img_name.endswith(".jpg"):
        return True


def     initialize_vertebrae_localization():
    global vertebraeLocalization_initialized
    global model
    if not vertebraeLocalization_initialized:
        rf = Roboflow(api_key="XXXX")
        project = rf.workspace().project(vertebraeLocalization_projectName)
        model = project.version(2).model
        vertebraeLocalization_initialized = True


def segment_vertebrae():
    global vertebraeLocations
    result = model.predict(input_imageName, confidence=40).json()
    print(result)
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    label_annotator = sv.LabelAnnotator()
    #bounding_box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    #print(detections.xyxy.astype(dtype=np.integer))
    xyxy_predictions = detections.xyxy.astype(dtype=np.integer)
    #print(xyxy_predictions)
    image = cv2.imread(input_imageName)
    back_up_image = cv2.imread(input_imageName)
    vertebrae_masks = image.copy()
    vertebrae_masks[:, :, :] = 0
    for detection_idx in np.flip(np.argsort(detections.area)):
        mask = detections.mask[detection_idx]
        vertebrae_masks[mask, :] = [0, 255, 0]
        image[mask, :] = [0, 255, 0]
        #break
    im = cv2.addWeighted(vertebrae_masks, 0.2, back_up_image, 0.8, 0)
    sv.plot_image(image=im, size=(16, 16))
    sv.plot_image(image=image, size=(16, 16))
    disc(image,im)


def disc(image, im):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV (narrow range around green)
    lower_green = np.array([55, 200, 200])  # Lower bound for green
    upper_green = np.array([65, 255, 255])  # Upper bound for green

    # Create a mask for the specified green color range
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there are at least two contours found
    if len(contours) >= 2:
        # Sort contours by their bounding box's top y-coordinate
        contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

        # Calculate the vertical distances between each pair of consecutive contours
        vertical_distances = []
        for i in range(len(contours) - 1):
            # Get the bounding boxes of the current and next contour
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            x2, y2, w2, h2 = cv2.boundingRect(contours[i + 1])

            # Calculate the vertical distance between the two bounding boxes
            vertical_distance = y2 - (y1 + h1)
            vertical_distances.append(vertical_distance)

            # Draw the contours and the vertical distance on the image for visualization
            cv2.drawContours(im, [contours[i]], -1, (0, 0, 255), 1)
            cv2.drawContours(im, [contours[i + 1]], -1, (0, 0, 255), 1)
            midpoint_x = (x1 + x2) // 2
            midpoint_y1 = y1 + h1
            midpoint_y2 = y2
            cv2.line(im, (midpoint_x, midpoint_y1), (midpoint_x, midpoint_y2), (255, 0, 0), 1)
            cv2.putText(im, f'{vertical_distance}px', (midpoint_x, (midpoint_y1 + midpoint_y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        cv2.imwrite('vertebrae_disc.jpg', im)
        sv.plot_image(im, size=(16, 16))
        for i, distance in enumerate(vertical_distances):
            print(f'Vertical distance between contour {i} and contour {i + 1}: {distance} pixels')
    else:
        print("Not enough contours found to fill the area between them.")


if __name__ == "__main__":
    initialize_vertebrae_localization()
    segment_vertebrae()
