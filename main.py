# from inference_sdk import InferenceHTTPClient
#
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="H3uNDksCMoK3AIQ7GAg7"
# )
#
# result = CLIENT.infer("Datasets/Composite Dataset of Lumbar Spine Mid-Sagittal Images with Annotations and Clinically Relevant Spinal Measurements/1. Images/T1_0001_S8.png", model_id="vertebra-segmentation/1")
# print(result.values())

from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="H3uNDksCMoK3AIQ7GAg7")
project = rf.workspace().project("vertebra-segmentation")
model = project.version(1).model
img_sample = "Datasets/Composite Dataset of Lumbar Spine Mid-Sagittal Images with Annotations and Clinically Relevant Spinal Measurements/1. Images/T1_0001_S8.png"


result = model.predict(img_sample, confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_roboflow(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread(img_sample)

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))