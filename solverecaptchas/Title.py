from googletrans import Translator

import setting
from solverecaptchas import TitleSimillar

title_list =[
    # yolov8에서 트럭은 버스로
    ['버스','bus','autobuses', 'autobús', 'buses','truck','트럭','trucks'],
    ['자동차','차','차량','car','automóviles', 'cars', 'car', 'coches', 'coche'],
    ['자전거','bicycle', 'bicycles', 'bicicletas', 'bici'],
    ['소화전','fire hydrant','fire_hydrant','fire_hydrants','a_fire_hydrant'],
    ['횡단보도','횡단 보도','crosswalk','crosswalks','cross_walks','cross_walk'],
    ['오토바이','motorcycle', 'motorcycles', 'motorbike', 'motorbikes', 'moto', 'motos'],
    ['신호등','traffic light', 'traffic lights', 'traffic_light', 'traffic_lights', 'semaforo', 'semaforos'],
    ['계단','stair', 'stairs', 'escaleras', 'escalera','Stair'],
    ['굴뚝','chimney', 'chimneys', 'chimenea'],
    ['교각','다리','bridge','bridges'],
    ['비행기','airplane', 'airplanes', 'aeroplane', 'aeroplanes', 'avión', 'aviones'],
    ['기차','train', 'trains', 'tren', 'trenes'],
    ['stop signs','stop sign','stop_sign','stop_signs','stop','street signs','street_signs','street_sign'],
    ['parking meters','parking meter','parking_meters','parking_meter'],
    ['의자','bench','benches','benchs','chair','chairs','silla','sillas']]

# 두 단어가 같은 그룹에 속하는지
def is_same_group(word1,word2):
    for group in title_list:
        if word1 in group and word2 in group:
            return True
    return False


# 단어가 속한 그룹
# 없는 단어의 경우 None
def get_title_group(word):
    for group in title_list:
        if word in group:
            return group
    return None

# 유사한 단어 추가
# 없으면 빈 리스트
def append_simillar_title(word):
    list_title_temp = []
    if setting.find_simillar_word:
        set_word = TitleSimillar.find_similar_words(word)
        list_title_temp = list(set_word)

    return list_title_temp


def is_title_korean(title):
    if title.isalpha():
        return False
    else:
        return True

def translate_english_to_korean(word):
    translator = Translator()
    translation = translator.translate(word, src='ko', dest='en')
    return translation.text

#
def get_first_english_title_in_group(group:list):
    for title in group:
        flag = True
        for ch in title:
            if not (ch.encode().isalpha() or ch == ' ' or ch == '_'):
                flag = False
                break
        if not flag:
            continue
        else:
            return title
    return None

if __name__ == '__main__':
    print("fire hydrant".encode().isalpha())
    pass
    # print(
    #     get_first_english_title_in_group())
    # a=translate_english_to_korean('차량')
    # print(TitleSimillar.find_similar_words(a))