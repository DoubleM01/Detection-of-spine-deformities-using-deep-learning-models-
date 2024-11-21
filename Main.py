import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import os
import argparse


def main(input_image, output_image):
    # Validate input file existence and format
    if not os.path.isfile(input_image):
        raise FileNotFoundError(f"The input file '{input_image}' does not exist.")
    if not input_image.lower().endswith(('.jpg', '.png')):
        raise ValueError(f"The input file '{input_image}' must be in .jpg or .png format.")

    # Ensure the output file has the correct format jpg | png
    if not output_image.lower().endswith(('.jpg', '.png')):
        raise ValueError(f"The output file '{output_image}' must be in .jpg or .png format.")

    process_image(input_image, output_image)

def process_image(input_path, output_path):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="XXXX"
    )
    image = input_path
    result = CLIENT.infer(image, model_id="fitikdeneme/3")
    print(result)
    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_roboflow(result)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    image = cv2.imread(image)

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    plt.imsave(output_path, annotated_image)
    sv.plot_image(image=annotated_image, size=(16, 16))

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Detect spine deformities using deep learning models.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the directory where output results will be saved."
    )

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(input_path=args.input, output_path=args.output)
