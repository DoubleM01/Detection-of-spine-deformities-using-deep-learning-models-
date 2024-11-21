import cv2

def convert_png_2jpg(image_name):
    image = cv2.imread(image_name)
    cv2.imwrite(image_name.replace(".png", ".jpg"), image)