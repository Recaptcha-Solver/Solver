from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import cv2
import numpy as np


def predict_on_image(model, img, conf):
    result = model(img, conf=conf)[0]

    # detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

    # segmentation
    masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W)
    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
    # rescale masks to original image
    masks = scale_image(masks, result.masks.orig_shape)
    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    return boxes, masks, cls, probs


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

# Load a model
model = YOLO("C:\\Users\\vvpas\\PycharmProjects\\recaptcha_solver\\yolov8\\yolov8x-seg.pt")

# load image by OpenCV like numpy.array
img = cv2.imread("C:\\Users\\vvpas\\PycharmProjects\\recaptcha_solver\\yolov8\\motor.jpg")

# predict by YOLOv8
boxes, masks, cls, probs = predict_on_image(model, img, conf=0.55)

# overlay masks on original image
image_with_masks = np.copy(img)
zero_array = np.zeros_like(image_with_masks)
for mask_i in masks:
    image_with_masks = overlay(zero_array, mask_i, color=(255,255,255), alpha=0.3)


# Saving the image
cv2.imwrite('frame_00001_with_masks.PNG', image_with_masks)