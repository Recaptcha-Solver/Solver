import json
import time

current_title = ""
start_time = 0
end_time = 0

result = []


def save_to_json():
    # result to json
    json_str = json.dumps(result, indent=4)
    print(f'saved json string : {json_str}')
    file_name = 'result_'+str(time.time())+'.json'
    with open(file_name, 'w') as json_file:
        json.dump(result, json_file, indent=4)


def save_to_list(title, start, end):
    result.append({'title': title, 'start': start, 'end': end})
    print("결과 수 :",len(result))


def start_timer(title):
    global start_time, current_title
    start_time = time.time()
    current_title = title
    return None


def end_timer():
    global start_time, end_time, current_title
    end_time = time.time()
    print(f"{current_title} 소요시간 : {end_time-start_time}초")
    save_to_list(current_title, start_time, end_time)
    start_time = 0
    end_time = 0
