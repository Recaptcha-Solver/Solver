# import tensorflow as tf
# import cv2
# import numpy as np
# import solverecaptchas.utils as utils
#
# import matplotlib.pyplot as plt
#
# from PIL import Image
#
#
# model_path = "C:\\Users\\vvpas\\PycharmProjects\\recaptcha_solver\\tensorflow_model\\saved.h5"
# image_path = "C:\\Users\\vvpas\\PycharmProjects\\recaptcha_solver\\tensorflow_model\\cross_real.jpg"
#
# # 예측할 이미지 로딩
# image = utils.load_img_to_96x96x1(image_path)
#
# # model = tf.saved_model.load(model_path) >> 그래프만 로드하는 방식 : pb파일이 들어 있는 "폴더" 경로
# model = tf.keras.models.load_model(model_path)
# model.summary()
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# # model.predict()
# # ['Bus','Car','Crosswalk','Hydrant','Mountain','Palm','Traffic Light']  # 7개
# classes = model.predict_splited(image)
# best_class = np.argmax(classes)
# print(f'클래스 예측 : {best_class}')
#
#
#
#
