import cv2
import os
import numpy as np
from PIL import Image


__all__ = [
    'is_marked',
    "get_output_layers",
    "draw_prediction",
    "predict"
]

import setting
import solverecaptchas.utils as utils
import yolov8.yolov8_predict as yolov8_predict
import yolov8.yolov8_image_utils as yolov8_image_utils
from solverecaptchas import Title


def is_marked(img_path):
    """Detect specific color for detect marked"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r == 0 and g == 0 and b == 254:  # Detect Blue Color
                return True
    return False


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, x, y, x_plus_w, y_plus_h):
    """Paint Rectangle Blue for detect prediction"""
    color = 256  # Blue
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, cv2.FILLED)

async def predict_yolov8_add_new(full_image_path,title_group,split_size):
    results = await yolov8_predict.predict_yolov8_add_new(full_image_path)
    selected = yolov8_image_utils.select_split_img(results,cv2.imread(full_image_path),title_group,split_size,setting.cv2_test_mode)
    return selected


# split_size : 16 또는 9
# full_image : 분할되지 않은 전체 사진
async def predict_yolov8(full_image_path,title_group,split_size):

    denoised_image_cv2, denoised_file_path = yolov8_image_utils.remove_noise(full_image_path)
    results = await yolov8_predict.predict_yolov8(full_image_path)

    selected_denoised = []
    # 3x3은 노이즈 이미지 추가
    if split_size == 9:
        if setting.use_denoise:
            results_denoised = await yolov8_predict.predict_yolov8(denoised_file_path)
            selected_denoised = yolov8_image_utils.select_split_img(results_denoised,denoised_image_cv2,title_group,split_size,setting.cv2_test_mode)

    selected_origin = yolov8_image_utils.select_split_img(results,cv2.imread(full_image_path),title_group,split_size,setting.cv2_test_mode)
    return selected_origin,selected_denoised

async def predict_yolov3(net, file,obj=None):
    """Predict Object on image"""
    file_names = setting.yolov3_txt_path
    image = cv2.imread(file)
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    conf_threshold = 0.5
    nms_threshold = 0.4

    with open(file_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    if obj is None:
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        classes_names = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    classes_names.append(classes[class_id])
        return classes_names  # Return all names object in the images
    else:
        out_path = f"pictures/tmp/{hash(file)}.jpg"
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        print(f"outs 리스트 : {outs}")

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        out = False
        for i in indices:
            if classes[int(class_ids[int(i)])] == obj or (obj == 'vehicles' and (
                    classes[int(class_ids[int(i)])] == 'car' or classes[int(class_ids[int(i)])] == 'truck')):
                out = out_path
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(image, round(x), round(y), round(x + w), round(y + h))
            # Save Image
        if out:
            cv2.imwrite(out_path, image)
        return out  # Return path of images or False if not found object
#
#
# async def predict_tensorflow???(file_path,title,tf_net):
#     threshold = 0.5 # 0.5 넘으면 포함된다고 인식
#     model = tf_net
#     image = utils.load_img_to_96x96x1(file_path)
#     classes = model.predict_splited(image)
#     # best_class = np.argmax(classes) # print(f'클래스 예측 : {best_class}')
#     tf_labels_ignore_case = list(map(str.lower ,setting.tf_labels)) # 소문자로
#
#     title_label_num = None # title이 Bus면 몇번 레이블이 Bus인지
#     if not (title in tf_labels_ignore_case):
#         print(f'{title}은 학습된 텐서플로우 모델에 존재하지 않는 레이블')
#         exit(1)
#     else:
#         title_label_num = tf_labels_ignore_case.index(title)
#
#     return_classes = []
#     for idx,predict_result in enumerate(classes[0]):
#         if predict_result >= threshold:
#             return_classes.append(tf_labels_ignore_case[idx])
#
#     return return_classes

# image_inst : Image 객체 (image.py)
# image_path : 분할 전 사진 경로
# return : 0~9 또는 0~16에서 선택된 번호 리스트
async def predict(image_inst, image_path, obj=None):
    # title = image_inst.title # 23.05.21 이후 사용 X

    # 리캡차에 나온 타이틀과 동일할 수 있는 모든 타이틀 그룹
    title_group = Title.get_title_group(image_inst.title)
    if title_group is None:
        print(f"\n\n------Title.py에 존재하지 않는 타이틀------\n{image_inst.title}")
        return list() # 빈 리스트 리턴


    yolov3_net = image_inst.yolov3_net
    tf_net = image_inst.tf_net

    split_size = image_inst.pieces
    # 텐서플로우
    selected_tensorflow = []
    if split_size == 9 and setting.use_tensorflow_7: # 텐서플로우 모델은 3x3에서만 사용
        selected_tensorflow = await predict_tensorflow(tf_net,image_inst,image_path,title_group,obj)

    # yolo8
    selected_yolov8_origin,selected_yolov8_denoised = await predict_yolov8(image_path,title_group,split_size) # image_inst.pieces : 16 또는 9

    # 횡단보도 + 기타
    selected_yolov8_add_new = await predict_yolov8_add_new(image_path,title_group,split_size)


    print(f"YOLO V8 select index : {selected_yolov8_origin}")
    print(f"New YOLO V8 select index : {selected_yolov8_add_new}")
    if setting.use_tensorflow_7:
        print(f"tensorflow select index : {selected_tensorflow}")
    if setting.use_denoise:
        print(f"(3x3) YOLO V8 denoised select index : {selected_yolov8_denoised}")

    # 두 결과를 합치기
    combined_result = list(set(selected_tensorflow+selected_yolov8_origin+selected_yolov8_denoised+selected_yolov8_add_new))
    return combined_result

# return : 0~8번 또는 0~15번에서 선택된 리스트
async def predict_tensorflow(tf_net, image_inst, image_path,title_group, obj=None):
    selected = []
    image_obj = Image.open(image_path)
    utils.split_image(image_obj, image_inst.pieces, image_inst.cur_image_path, image_inst.image_hash)
    for i in range(image_inst.pieces):
        # 원본(yolov3) : result = await predict_splited(image_inst.net, image_inst.tf_net, os.path.join(image_inst.cur_image_path, f'{image_inst.image_hash}_{i}.jpg'), image_inst.title)
        result_tf = await predict_tensorflow_splited(image_inst.tf_net,
                                                     os.path.join(image_inst.cur_image_path, f'{image_inst.image_hash}_{i}.jpg'),
                                                     title_group)
        if result_tf:
            selected.append(i)

    return selected


# predict_splited()함수는 이미 쪼개진 사진 한장에서 고르는 것
# 텐서플로우 모델은 분할된 한장의 사진으로 예측
# yolov8 등은 전체 사진으로 체크하므로 이 함수를 호출할 필요 X
# return : 해당 title이 포함 되었는지 (True / False)
async def predict_tensorflow_splited(tf_net, file_path, title_group, obj=None):
    # title = title.lower()
    tf_labels = list(map(str.lower, setting.tf_model_label_predict))
    threshold = 0.5  # 0.5 넘으면 포함된다고 인식
    model = tf_net
    image = utils.load_img_to_96x96x1(file_path)

    tf_labels_set = set(tf_labels)
    title_group_set = set(title_group)
    common_set = tf_labels_set.intersection(title_group_set)
    if len(common_set) > 0: # 레이블 존재하는 경우
        classes = model.predict(image)
        # best_class = np.argmax(classes) # print(f'클래스 예측 : {best_class}')
        tf_labels_ignore_case = list(map(str.lower, setting.tf_labels))  # 소문자로
        # title_label_num = tf_labels_ignore_case.index(title)  # title이 Bus면 몇번 레이블이 Bus인지

        clss_tf = []
        for idx, predict_result in enumerate(classes[0]):
            if predict_result >= threshold:
                clss_tf.append(tf_labels_ignore_case[idx])

        # 한장의 예측 결과에 원하는 레이블이 있는 경우
        for label in common_set:
            if label in clss_tf:
                return True
            else:
                return False
    else:
        return False


