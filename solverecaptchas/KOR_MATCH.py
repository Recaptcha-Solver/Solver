
kor_dict = {"버스":"bus","소화전":"fire hydrant",'오토바이':'motorcycles','신호등':'traffic light','자동차':'car'
            ,'자전거':'bicycle','횡단보도':'crosswalk'
            ,'계단':'stair','교각':'bridge'}


def convert_to_english(title):
    global kor_dict
    if title in kor_dict:
        return kor_dict[title]
    else:
        print(f"존재 하지 않는 한글 키 : {title}")
        # 한글 경로를 읽지 못하기 때문에 스킵
        return None
