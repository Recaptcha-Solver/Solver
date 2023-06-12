#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import tempfile
import time
import setting

import solverecaptchas.utils as utils
import solverecaptchas.image as image

from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

from urllib.parse import urlparse

from solverecaptchas import result_save
from yolov8 import yolov8_predict


class Solver(object):

    def load_tf_model(self):
        import tensorflow as tf
        model = tf.keras.models.load_model(setting.tf_model_path)
        model.summary()
        model.compile(optimizer=setting.tf_optimizer,
                      loss=setting.tf_loss_function,
                      metrics=['accuracy'])
        return model

    async def start_test(self,img_path,image_class):
        from solverecaptchas.predict import predict_tensorflow_splited
        im = image.SolveImage(page=None, image_frame=None, net=self.yolov3_net, tf_net=self.tf_net)
        result = await predict_tensorflow_splited(self.tf_net, img_path, image_class)
        print("테스트 결과")
        print(result)

    # 이미지 테스트
    def __init__(self,
        pageurl, sitekey, proxy=None, headless=False, timeout=30000*1000,
            solve_by_image=True, solve_by_audio=False):

        if setting.use_yolov3:
            self.yolov3_net = cv2.dnn.readNet(setting.yolov3_weights, setting.yolov3_cfg)
        else:
            self.yolov3_net = None

        if setting.use_audio_model:
            import shutil
            from vosk import Model
            model = Model('model') # Vosk 모델 로드 (음성 캡차)
        else:
            model = None

        self.pageurl = pageurl
        self.sitekey = sitekey
        self.proxy = proxy
        self.headless = headless
        self.timeout = timeout
        self.model = model
        self.solve_by_image = solve_by_image
        self.solve_by_audio = solve_by_audio
        if setting.use_tensorflow_7:
            self.tf_net = self.load_tf_model()
        else:
            self.tf_net = None

        print("load yolov8 model..")
        import yolov8.yolov8_predict
        yolov8_predict.preheat_yolov8()


    async def click_boj_login(self):
        id_text_field = await self.page.wait_for_selector("#login_form > div:nth-child(2) > input")
        password_text_field = await self.page.wait_for_selector("#login_form > div:nth-child(3) > input")
        await id_text_field.type("kys1130")
        await password_text_field.type("qwer1234")
        submit_button = await self.page.wait_for_selector("#submit_button")
        await submit_button.click()
        await password_text_field.click()

    async def start_all_page(self,audio=False):
        self.browser = await self.get_browser_all()
        self.page = await self.new_page()
        await self.apply_stealth()
        # await self.reroute_requests()
        await self.goto_page()

        # 다섯번 재시도
        for retry_time in range(5):
            try:
                await self.get_frames_new()
                # await self.get_frames()  # 여기서 예외 발생하면 디버그 찍고 Step Over로 넘기기 (로딩되기 전에 먼저 호출되면 오류나는듯함)
                break
            except:
                print("\nget_frames 예외 발생 - 재시도")
                continue

        # await self.click_boj_login()
        await self.click_checkbox() # 리캡차 데모 전용
        if not audio:
            result = await self.solve_image()
        else:
            result = await self.solve_audio()

        # await self.cleanup()
        return result


    async def start(self):
        self.browser = await self.get_browser()
        # self.browser = await self.get_browser_all()
        self.page = await self.new_page()
        await self.apply_stealth()
        await self.reroute_requests()
        await self.goto_page()
        await self.get_frames() # 여기서 예외 발생하면 디버그 찍고 Step Over로 넘기기 (로딩되기 전에 먼저 호출되면 오류나는듯함)
        await self.click_checkbox()
        if self.solve_by_image:
            result = await self.solve_image()
        else:
            result = await self.solve_audio()
        await self.cleanup()
        if result:
            return result

    async def get_browser_all(self):
        playwright = await async_playwright().start()
        # browser = await playwright.webkit.launch(
        #     headless=False,
        #     proxy=None
        # )
        browser = await playwright.chromium.launch(channel="chrome", headless=False, proxy=None)
        if self.proxy:
            self.proxy = {'server': self.proxy}
        # browser = await playwright.webkit.launch(
        #     headless=self.headless,
        #     proxy=self.proxy,
        # )
        return browser

    async def get_browser(self):
        playwright = await async_playwright().start()
        if self.proxy:
            self.proxy = {'server': self.proxy}
        browser = await playwright.webkit.launch(
            headless=self.headless,
            proxy=self.proxy,
        )
        return browser
    
    async def new_page(self):
        page = await self.browser.new_page()
        page.set_default_timeout(self.timeout)
        return page

    async def apply_stealth(self):
        await stealth_async(self.page)

    # 이 부분에서 site key가 필요 (없어도 문제 X)
    async def reroute_requests(self):
        parsed_url = urlparse(self.pageurl)
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        await self.page.route(f"{scheme}://{netloc}/*", lambda route: 
            route.fulfill(
            content_type="text/html",
            body=f"<script src=https://www.google.com/recaptcha/api.js?hl=en><"
                 "/script>"
                 f"<div class=g-recaptcha data-sitekey={self.sitekey}></div>")
        )

    async def goto_page(self):
        await self.page.goto(self.pageurl,wait_until="commit")
        await self.page.wait_for_load_state("load")

    # 리캡차 "로봇이 아닙니다" 클릭하는 체크박스
    async def click_checkbox(self):
        checkbox = await self.checkbox_frame.wait_for_selector(
            "#recaptcha-anchor")
        await checkbox.click()

    # 엔터프라이즈 버전
    async def get_frames_new(self):
        # exists = self.page.evaluate(f'document.querySelector("enterprise") !== null')
        # exists = self.page.evaluate("() => { return typeof yourDirectoryPath !== 'undefined'; }")
        is_enterprise = True
        for frame in self.page.frames:
            if 'api2/anchor?' in frame.url:
                is_enterprise = False
                break

        if is_enterprise:
            print("enterprise 버전")
            await self.page.wait_for_selector("iframe[src*=\"enterprise/anchor\"]",
                                              state='attached')
            self.checkbox_frame = next(frame for frame in self.page.frames
                                       if "enterprise/anchor" in frame.url)
            await self.page.wait_for_selector("iframe[src*=\"enterprise/bframe\"]",
                                              state='attached')
            self.image_frame = next(frame for frame in self.page.frames
                                    if "enterprise/bframe" in frame.url)
        else:
            print("데모 버전")
            await self.page.wait_for_selector("iframe[src*=\"api2/anchor\"]",
                                              state='attached')
            self.checkbox_frame = next(frame for frame in self.page.frames
                                       if "api2/anchor" in frame.url)
            await self.page.wait_for_selector("iframe[src*=\"api2/bframe\"]",
                                              state='attached')
            self.image_frame = next(frame for frame in self.page.frames
                                    if "api2/bframe" in frame.url)


        # async with async_playwright() as playwright:
        #     # Check if the directory exists by evaluating a JavaScript expression
        #     exists = self.page.evaluate(f'document.querySelector("enterprise") !== null')
        #     return exists

    async def get_frames(self):
        await self.page.wait_for_selector("iframe[src*=\"api2/anchor\"]",
            state='attached')
        self.checkbox_frame = next(frame for frame in self.page.frames 
            if "api2/anchor" in frame.url)
        await self.page.wait_for_selector("iframe[src*=\"api2/bframe\"]",
            state='attached')
        self.image_frame = next(frame for frame in self.page.frames 
            if "api2/bframe" in frame.url)

    async def click_audio_button_new(self):
        audio_button = await self.image_frame.wait_for_selector("#recaptcha-audio-button")
        await audio_button.click()


    async def click_audio_button(self):
        audio_button = await self.image_frame.wait_for_selector("#recaptcha-au"
            "dio-button")
        await audio_button.click()

    async def check_detection(self, timeout):
        timeout = time.time() + timeout
        while time.time() < timeout:
            content = await self.image_frame.content()
            if 'Try again later' in content:
                return 'detected'
            elif 'Press PLAY to listen' in content:
                return 'solve'
            else:
                result = await self.page.evaluate('document.getElementById("g-'
                    'recaptcha-response").value !== ""')
                if result:
                    return result

    async def solve_audio(self):
        import shutil
        await self.click_audio_button_new()
        while 1:
            if result == 'solve':
                play_button = await self.image_frame.wait_for_selector("#audio-source",
                    state="attached")
                audio_source = await play_button.evaluate("node => node.src")
                audio_data = await utils.get_page(audio_source, binary=True)
                tmpd = tempfile.mkdtemp()
                tmpf = os.path.join(tmpd, "audio.mp3")
                await utils.save_file(tmpf, data=audio_data, binary=True)
                audio_response = await utils.get_text(tmpf)
                audio_response_input = await self.image_frame.wait_for_selector("#audi"
                    "o-response", state="attached")
                await audio_response_input.fill(audio_response['text'])
                recaptcha_verify_button = await self.image_frame.wait_for_selector("#r"
                    "ecaptcha-verify-button", state="attached")
                await recaptcha_verify_button.click()
                shutil.rmtree(tmpd)
                result = await self.get_recaptcha_response()
                if result:
                    return result
            else:
                break

    async def solve_image(self):
        im = image.SolveImage(self.page, self.image_frame, yolov3_net=self.yolov3_net, tf_net=self.tf_net)
        await im.solve_by_image(solver_inst=self)
        result = await self.get_recaptcha_response()
        result_save.save_to_json()
        if result:
            return result

    async def get_recaptcha_response(self):
        if await self.page.evaluate('document.getElementById("g-recaptcha-resp'
            'onse").value !== ""'):
            recaptcha_response = await self.page.evaluate('document.getElementById'
                '("g-recaptcha-response").value')
            return recaptcha_response

    async def cleanup(self):
        await self.browser.close()
