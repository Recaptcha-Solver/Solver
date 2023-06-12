import aiohttp
import cv2
import aiofiles
import asyncio
import itertools
import json
import numpy as np
import os
import requests
import sys
import wave

from functools import partial, wraps
from pydub import AudioSegment
from vosk import KaldiRecognizer, SetLogLevel

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

SetLogLevel(-1)

def threaded(func):
    @wraps(func)
    async def wrap(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))
    return wrap

def mp3_to_wav(mp3_filename):
    wav_filename = mp3_filename.replace(".mp3", ".wav")
    segment = AudioSegment.from_mp3(mp3_filename)
    sound = segment.set_channels(1).set_frame_rate(16000)
    sound.export(wav_filename, format="wav")
    return wav_filename

@threaded
def get_text(mp3_filename, model):
    wav_filename = mp3_to_wav(mp3_filename)
    wf = wave.open(wav_filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    return json.loads(rec.FinalResult())

async def save_file(file, data, binary=False):
    mode = "w" if not binary else "wb"
    async with aiofiles.open(file, mode=mode) as f:
        await f.write(data)

async def load_file(file, binary=False):
    mode = "r" if not binary else "rb"
    async with aiofiles.open(file, mode=mode) as f:
        return await f.read()

def load_img_to_96x96x1(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 그레이 스케일로 바로 읽기
    image = cv2.resize(image, dsize=(96, 96))  # 크기 조절절
    image = np.expand_dims(image, axis=2)  # (96,96) to (96,96,1) -> 차원 추가

    # cv2.imshow("test",image)
    # cv2.waitKey(0)
    return_x = np.zeros(shape=(1, 96, 96, 1)) # 맨 왼쪽이 1인 이유는 이미지 한개라서. 이미지 여러개 하면 맨 앞 1을 수정
    return_x[0] = image
    return return_x

@threaded
def get_page_win(
        url,
        proxy=None,
        proxy_auth=None,
        binary=False,
        timeout=300):
    proxies = None
    if proxy:
        if proxy_auth:
            proxy = proxy.replace("http://", "")
            username = proxy_auth['username']
            password = proxy_auth['password']
            proxies = {
                "http": f"http://{username}:{password}@{proxy}",
                "https": f"http://{username}:{password}@{proxy}"}
        else:
            proxies = {"http": proxy, "https": proxy}
    with requests.Session() as session:
        response = session.get(
            url,
            proxies=proxies,
            timeout=timeout)
        if binary:
            return response.content
        return response.text


# 텍스트에 한글 포함 여부
def is_korean(text):
    for character in text:
        if ord('가') <= ord(character) <= ord('힣'):
            return True
    return False


async def get_page(
        url,
        proxy=None,
        proxy_auth=None,
        binary=False,
        timeout=300):
    if sys.platform == "win32":
        # SSL Doesn't work on aiohttp through ProactorLoop so we use Requests
        return await get_page_win(
            url, proxy, proxy_auth, binary, timeout)
    else:
        if proxy_auth:
            proxy_auth = aiohttp.BasicAuth(
                proxy_auth['username'], proxy_auth['password'])
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    url,
                    proxy=proxy,
                    proxy_auth=proxy_auth,
                    timeout=timeout) as response:
                if binary:
                    return await response.read()
                return await response.text()

def split_image(image_obj, pieces, save_to, name):
    """Splits an image into constituent pictures of x"""
    width, height = image_obj.size
    row_length = int(np.sqrt(pieces))
    interval = width // row_length
    for x, y in itertools.product(range(row_length), repeat=2):
        cropped = image_obj.crop((interval * x, interval * y, interval * (x + 1), interval * (y + 1)))
        cropped.save(os.path.join(save_to, f'{name}_{y * row_length + x}.jpg'))
