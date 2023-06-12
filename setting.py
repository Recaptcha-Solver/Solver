import configparser
import os.path

import setting

"""-
< 구글 데모 >
pageurl = 'https://www.google.com/recaptcha/api2/demo'
sitekey = '6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-'
"""


def make_config():
    config = configparser.ConfigParser()
    config['solve_page'] = {}
    config['solve_page']['pageurl'] = 'https://www.google.com/recaptcha/api2/demo' # 구글 데모 사이트
    config['solve_page']['sitekey'] = '6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-' # 구글 데모 사이트

    config['base_link'] = {}
    config['base_link']['link'] = 'C:\\~~~'
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)

def get_config():
    # config 없으면 생성
    if not os.path.exists('config.ini'):
        make_config()

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    pageurl = config['solve_page']['pageurl']
    sitekey = config['solve_page']['sitekey']
    base_link = config['base_link']['link']


    return pageurl,sitekey,base_link


pageurl,sitekey,base_link = get_config()

base_link_model = base_link+"\\model"

cv2_test_mode = False # 예측하면서 이미지 화면에 띄울건지


# 음성 모델 사용 여부
use_audio_model = False

# 노이즈 자동 제거 사용
use_denoise = False

# 텐서플로우 모델
use_tensorflow_7 = False
tf_model_label_predict = {'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Mountain', 'Palm', 'Traffic Light'} # yolov3 대신 텐서플로우 모델로 예측할 레이블
tf_labels = ['Bus','Car','Crosswalk','Hydrant','Mountain','Palm','Traffic Light'] # 학습된 모델 레이블 순서
tf_model_path = f'{base_link}\\tensorflow_model\\saved.h5'  # 텐서플로우 모델
tf_optimizer = 'adam'
tf_loss_function = 'sparse_categorical_crossentropy'

# yolov8 모델
yolov8_model_dir_path = base_link+"\\yolov8"

# yolo v3 모델 정보
use_yolov3 = False
yolov3_txt_path = base_link_model + '\\yolov3.txt'
yolov3_weights = base_link_model + '\\yolov3.weights'
yolov3_cfg = base_link_model + '\\yolov3.cfg'


# 없는 단어가 나왔을 때 자동으로 비슷한 단어 목록 찾기
find_simillar_word = True




