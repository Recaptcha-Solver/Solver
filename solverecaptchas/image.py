#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Image solving module. """
import asyncio
import os
import random
import time
import cv2
import numpy as np
import io
import asyncio
from playwright.async_api import Playwright, async_playwright
import setting
from solverecaptchas import KOR_MATCH, predict, Title,package_dir,result_save
from solverecaptchas.predict import predict_tensorflow_splited
from PIL import Image
import solverecaptchas.utils as utils
import yolov8.yolov8_predict as yolov8_predict
import yolov8.yolov8_image_utils as yolov8_image_utils


class SolveImage():
    title = None
    pieces = None
    download = None
    cur_image_path = None

    def __init__(self, page, image_frame, yolov3_net, tf_net,proxy=None, proxy_auth=None, **kwargs):
        self.page = page
        self.image_frame = image_frame
        self.yolov3_net = yolov3_net
        self.tf_net = tf_net
        self.proxy=proxy
        self.proxy_auth=proxy_auth

    # 맨 처음이랑 사진 바뀔 때 다시 호출되는 함수
    async def get_start_data(self):
        """Detect pieces and get title image"""
        await self.get_title()
        image = await self.download_image() # image : file path

        await self.create_folder(self.title, image)
        file_path = os.path.join(self.cur_image_path, f'{self.image_hash}_{self.title}.jpg')
        await utils.save_file(file_path, image, binary=True)
        self.pieces = await self.image_no()
        return file_path

    async def solve_by_image(self,solver_inst=None):
        """Go through procedures to solve image"""
        while True:
            result = await self.check_detection(3)
            if result:
                break
            file_path = await self.get_start_data()
            result_save.start_timer(self.title)
            print(f"(while문) 새로운 이미지 다운로드 : title {self.title}")
            if self.pieces == 16:
                # choices = await self.choose(file_path) 230501 이후로 choose는 사용 X
                # choices = await predict.predict(self,self.cur_image_path,None)
                choices = await predict.predict(self,file_path,None)
                print(f"선택 번호 : {choices}")
                await self.click_image(choices,cnt=self.pieces)
                if choices: # 테스트 필요
                    await self.click_verify()
                    # url = self.image_frame.url
                    # print(f'url : {url}')
                    # print("테스트1 : click verify 이후 2초 휴식")
                    time.sleep(2)
                    title_test = await self.image_frame.wait_for_selector(
                        "#rc-imageselect > div.rc-imageselect-payload > div.rc-imageselect-instructions > div.rc-imageselect-desc-wrapper > div > strong")
                    print(f"타이틀 테스트1 : {await title_test.inner_text()}")
                    if not await self.is_next() and not await self.is_finish():
                        print("is_next, is_finish 둘 다 아님 -> reload")
                        await self.click_reload_button()
                    else:
                        result_save.end_timer()
                else:
                    print("이미지에서 선택된 부분 X -> reload")
                    await self.click_reload_button()
            elif self.pieces == 9:
                # 원본 : choices = await self.choose(file_path)
                # choices = await predict.predict(self, self.cur_image_path, None)
                # await self.images_test()
                choices = await predict.predict(self, file_path, None)
                print(f"선택 번호 : {choices}")
                await self.click_image(choices,cnt=self.pieces)
                # url = self.image_frame.url
                # print(f'url : {url}')
                if choices:
                    if await self.is_one_selected():
                        # 원본 :  await self.cycle_selected(choices) # 3x3만 해당
                        # before_hashes : 새로운 사진이 로딩되지 않은 상태에서 사진을 다운받아서 동일한 사진이 된 경우
                        before_hashes = dict()
                        full_hash = yolov8_image_utils.get_image_hash(file_path)
                        for before_choice in choices:
                            before_hashes[before_choice] = full_hash # 첫 이미지는 9개 이미지 모두 FULL Image이므로 다 동일한 해시값
                        await self.new_cycle_selected(choices,before_hashes,self.pieces) # 3x3만 해당
                        await self.click_verify()
                        print("테스트2 : click verify 이후 2초 휴식")
                        time.sleep(2)
                        # title_test = await self.image_frame.wait_for_selector(
                        #     "#rc-imageselect > div.rc-imageselect-payload > div.rc-imageselect-instructions > div.rc-imageselect-desc-wrapper > div > strong")
                        # print(f"타이틀 테스트2 : {await title_test.inner_text()}")
                        if not await self.is_next() and not await self.is_finish():
                            print("is_next, is_finish 둘 다 아님 -> reload")
                            await self.click_reload_button()
                        else:
                            result_save.end_timer()
                    else:
                        await self.click_verify()
                        print("테스트3 : click verify 이후 2초 휴식")
                        time.sleep(2)
                        # title_test = await self.image_frame.wait_for_selector(
                        #     "#rc-imageselect > div.rc-imageselect-payload > div.rc-imageselect-instructions > div.rc-imageselect-desc-wrapper > div > strong")
                        # print(f"타이틀 테스트3 : {await title_test.inner_text()}")
                        if not await self.is_next() and not await self.is_finish():
                            print("is_next, is_finish 둘 다 아님 -> reload")
                            await self.click_reload_button()
                    result_save.end_timer()
                else:
                    await self.click_reload_button()



    async def images_test(self):
        jscode = (
            'document.getElementsByClassName("rc-image-tile-wrapper").length'
        )
        all_urls=[]
        await self.check_detection(3)
        cnt = await self.image_frame.evaluate(jscode)
        for i in range(cnt):
            js2 = (
                'document.getElementsByClassName("rc-image-tile-wrapper")['
                f'{i}].getElementsByTagName("img").length'
            )
            cnt2 = await self.image_frame.evaluate(js2)
            for j in range(cnt2):
                js3 = (
                    'document.getElementsByClassName("rc-image-tile-wrapper")['
                    f'{i}].getElementsByTagName("img")[{j}].src'
                )
                url = await self.image_frame.evaluate(js3)
                all_urls.append(url)

        for idx,url in enumerate(all_urls):
            image = await utils.get_page(url, self.proxy, self.proxy_auth, binary=True)
            await utils.save_file(f'{idx}_img.jpg', image, binary=True)


    async def new_cycle_selected(self, before_selected,before_hashes_dict,cnt):
        if len(before_selected) != len(before_hashes_dict):
            print("length not same")
            exit(1)


        while True:
            retry_download = set(before_selected)
            await self.check_detection(3)

            # 이미지가 로딩되는데 시간이 걸리므로 이미지 해시 값을 이전 이미지와 비교해서 새로운 이미지가 로딩 되었는지 확인
            while True: # 모든 이미지가 새로 로딩 될 때까지 시도
                for i in range(9):
                    if i in retry_download:
                        js_code = (
                            'document.getElementsByClassName("rc-image-tile-wrapper")['
                            f'{i}].getElementsByTagName("img")[0].src'
                        )
                        new_image_url = await self.image_frame.evaluate(js_code)
                        image = await utils.get_page(new_image_url, self.proxy, self.proxy_auth, binary=True)
                        new_image_local_path = f'{setting.base_link}\\pictures\\new_{i}_img.jpg'
                        await utils.save_file(new_image_local_path, image, binary=True)
                        # 해시값 동일 : 아직 새로운 사진 로딩 X
                        check_hash_value = yolov8_image_utils.get_image_hash(new_image_local_path)
                        if before_hashes_dict.get(i) == check_hash_value:
                            continue
                        else: # 다른 이미지인 경우
                            retry_download.remove(i)
                            before_hashes_dict[i] = check_hash_value
                if len(retry_download) > 0:
                    continue
                else:
                    break

            # 모든 이미지가 새로 다운로드 된 경우
            yolov8_image_utils.combine_image(before_selected)
            new_selected = await predict.predict(self,f'{setting.base_link}\\pictures\\combined.jpg')
            if len(new_selected) == 0:
                break
            else:
                random.shuffle(new_selected) # 랜덤으로 섞어서 순서 바꾸기 (3x3한정)
                await self.click_image(new_selected,cnt)

                # 새로 선택할 이미지만 해시값 복사
                # (이전 선택지에는 있었는데 새로운 선택지에는 없는 경우)
                new_selected_hashes = dict()
                for idx in before_hashes_dict.keys():
                    if idx in new_selected:
                        new_selected_hashes[idx] = before_hashes_dict[idx]
                before_selected = new_selected
                before_hashes_dict = new_selected_hashes
                continue






    # 사용 X (new_cycle_selected)
    async def cycle_selected(self, selected):
        import warnings
        warnings.warn("not use later")

        """Cyclic image selector"""
        while True:
            await self.check_detection(3)
            images = await self.get_images_block(selected)
            new_selected = []
            i = 0
            for image_url in images:
                if images != self.download:
                    image = await utils.get_page(image_url, self.proxy, self.proxy_auth, binary=True)
                    await self.create_folder(self.title, image)
                    file_path = os.path.join(
                        self.cur_image_path, f'{self.image_hash}_{self.title}.jpg')
                    # file_path에 저장되는 이미지는 이미 분할된 이미지
                    await utils.save_file(file_path, image, binary=True)
                    result = await predict_tensorflow_splited(self.tf_net, file_path, self.title)
                    if self.title == 'vehicles':
                        if 'car' in result or 'truck' in result or 'bus' in result:
                            new_selected.append(selected[i])
                    if self.title == 'motorcycles':
                        if 'car' in result or 'bicycle' in result or 'motorcycles' in result:
                            new_selected.append(selected[i])
                    if (self.title != 'vehicles' 
                            and self.title.replace('_', ' ') in result):
                        new_selected.append(selected[i])
                i += 1
            if new_selected:
                await self.click_image(new_selected)
            else:
                break


    async def click_verify(self):
        time.sleep(0.7)
        print("click verify -> 여기 다음줄에서 다음화면으로 넘어갔는데 is_next()가 true가 아니면 문제있는 것")
        # new_cycle하고나서 다음으로 넘어갔는데 is_next()가 true로 나오는 문제가 있음;
        await self.image_frame.locator('#recaptcha-verify-button').click()


    async def click_reload_button(self):
        print("click reload button")
        await self.image_frame.locator("#recaptcha-reload-button").click()


    # predict 및 선택된 위치 리턴
    # async def choose(self, image_path):
    #     """Get list of images selected"""
    #     selected = []
    #     if self.pieces == 9:
    #         image_obj = Image.open(image_path)
    #         utils.split_image(image_obj, self.pieces, self.cur_image_path, self.image_hash)
    #         for i in range(self.pieces):
    #             result = await predict_splited(self.yolov3_net, self.tf_net, os.path.join(self.cur_image_path, f'{self.image_hash}_{i}.jpg'), self.title)
    #             if self.title.replace('_', ' ') in result:
    #                 selected.append(i)
    #         os.remove(image_path) # selected[] : 9개의 이미지 중에서 선택된 이미지
    #     elif self.pieces == 16: # 전체를 좌표만 분할해서 판단 (23-04-26 추가)
    #         image_obj = Image.open(image_path) # 한장의 전체 이미지
    #         numpy_image = np.array(image_obj)
    #         results_yolov8 = await yolov8_predict.predict_yolov8(image_path)
    #         selected = yolov8_image_utils.select_split_img(results_yolov8,numpy_image,16,self.title)
    #
    #     #else:
    #     #    result = await predict(
    #     #        self.net, image_path, self.title.replace('_', ' '))
    #     #    if result is not False:
    #     #        image_obj = Image.open(result)
    #     #        utils.split_image(image_obj, self.pieces, self.cur_image_path)
    #     #        for i in range(self.pieces):
    #     #            if is_marked(f"{self.cur_image_path}/{i}.jpg"):
    #     #                selected.append(i)
    #     return selected

    async def get_images(self):
        """Get list of images"""
        table = self.image_frame.locator('table')
        rows = table.locator('tr')
        count = await rows.count()
        for i in range(count):
            cells = rows.nth(i).locator('td')
            count = await cells.count()
            for i in range(count):
                yield cells.nth(i)

    async def click_image(self, list_id:list, cnt):
        """Click specific images of the list"""

        # 9 : list_id 랜덤으로 섞어서 선택
        # 16 : 클릭 하는 시간 간격을 랜덤으로 설정

        print("click image")
        elements = self.image_frame.locator('.rc-imageselect-tile')
        if cnt == 9:
            random.shuffle(list_id) # 클릭 순서 랜덤으로
            for i in list_id:
                time.sleep(random.uniform(0.4, 0.9))
                await elements.nth(i).click()
        elif cnt == 16:
            for i in list_id:
                # sleep random time
                time.sleep(random.uniform(0.4, 0.9))
                await elements.nth(i).click()
        else:
            print("cnt error")
            return



    async def search_title(self, title):
        # 23.05.22 이후 사용 X
        """Search title with classes"""
        classes = ('bus', 'car', 'bicycle', 'fire_hydrant', 'crosswalk', 'stair', 'bridge', 'traffic_light',
                   'vehicles', 'motorbike', 'boat', 'chimneys')
        # Only English and Spanish detected!
        possible_titles = (
            ('autobuses', 'autobús', 'bus', 'buses'),
            ('automóviles', 'cars', 'car', 'coches', 'coche'),
            ('bicicletas', 'bicycles', 'bicycle', 'bici'),
            ('boca de incendios', 'boca_de_incendios', 'una_boca_de_incendios', 'fire_hydrant', 'fire_hydrants',
             'a_fire_hydrant', 'bocas_de_incendios'),
            ('cruces_peatonales', 'crosswalk', 'crosswalks', 'cross_walks', 'cross_walk', 'pasos_de_peatones'),
            ('escaleras', 'stair', 'stairs'),
            ('puentes', 'bridge', 'bridges'),
            ('semaforos', 'semaphore', 'semaphores', 'traffic_lights', 'traffic_light', 'semáforos'),
            ('vehículos', 'vehicles'),
            ('motocicletas', 'motocicleta', 'motorcycle','motorcycles', 'motorbike'),
            ('boat', 'boats', 'barcos', 'barco'),
            ('chimeneas', 'chimneys', 'chimney', 'chimenea')
        )
        i = 0
        for objects in possible_titles:
            if title in objects:
                return classes[i]
            i += 1
        return title

    async def pictures_of(self):
        """Get title of solve object"""
        el = await self.get_description_element()
        return str(el).replace(' ', '_')

    async def get_description_element(self):
        """Get text of object"""
        name = self.image_frame.locator('.rc-imageselect-desc-wrapper strong')
        return await name.inner_text()

    async def create_folder(self, title, image):
        """Create tmp folder and save image"""
        self.image_hash = hash(image)
        if not os.path.exists('pictures'):
            os.mkdir('pictures')
        if not os.path.exists(os.path.join('pictures', f'{title}')):
            os.mkdir(os.path.join('pictures', f'{title}'))
        if not os.path.exists(os.path.join('pictures', 'tmp')):
            os.mkdir(os.path.join('pictures', 'tmp'))
        self.cur_image_path = os.path.join(os.path.join('pictures', f'{title}'))
        if not os.path.exists(self.cur_image_path):
            os.mkdir(self.cur_image_path)

    async def get_image_url(self):
        """Get image url for download"""

        code = (
            'document.getElementsByClassName("rc-image-tile-wrapper")[0].'
            'getElementsByTagName("img")[0].src'
        )

        image_url = await self.image_frame.evaluate(code)
        return image_url

    async def image_no(self):
        """Get number of images in captcha"""
        return len([i async for i in self.get_images()])

    async def is_one_selected(self):
        print("is_one_selected")
        """Is one selection or multi-selection images"""
        code = (
            'document.getElementsByClassName("rc-imageselect-tileselected").'
            'length === 0'
        )
        ev = await self.image_frame.evaluate(code)
        return ev

    async def is_finish(self):
        """Return true if process is finish"""
        result = await self.check_detection(5)
        if result:
            print("is_finish : True")
            return True
        else:
            print("is_finish : False")
            return False

    async def is_next(self):
        """Verify if next captcha or the same"""
        image_url = await self.get_image_url()
        rt = False if image_url == self.download else True
        if not rt:
            return False
        else:
            # is_next()가 참이여도 잘못된 경우 존재

            # 밑에서 display가 none이 아닌 것이 있으면 실패
            # 해당되는 이미지를 모두 선택해주세요 메시지 -> 실패
            selector_more = "#rc-imageselect > div.rc-imageselect-payload > div:nth-child(4) > div.rc-imageselect-error-select-more"
            # 새 이미지도 확인해보세요
            selector_new_img = "#rc-imageselect > div.rc-imageselect-payload > div:nth-child(4) > div.rc-imageselect-error-dynamic-more"
            time.sleep(0.4)
            v1 = await self.image_frame.query_selector(selector_more)
            v2 = await self.image_frame.query_selector(selector_new_img)
            if v1 is not None or v2 is not None:
                tmp1 = await v1.get_attribute("style")
                tmp2 = await v2.get_attribute("style")
                if tmp1 != "display: none;" or tmp2 != "display: none;": # 새 이미지도 확인해보세요 등이 뜨는 경우
                    return False
        return rt

    # return : 다운로드 받은 후 로컬 이미지 경로
    async def download_image(self):
        """Download image captcha"""
        print("download_image")
        self.download = await self.get_image_url()
        return await utils.get_page(self.download, self.proxy, self.proxy_auth, binary=True)

    # 3x3 cycle 방식에서만 불리는 함수
    # 쪼개진 사진 다운 받는 링크
    async def get_images_block(self, images):
        """Get specific image in the block"""
        images_url = []
        for element in images:
            image_url = (
                'document.getElementsByClassName("rc-image-tile-wrapper")['
                f'{element}].getElementsByTagName("img")[0].src'
            )
            result = await self.image_frame.evaluate(image_url)
            images_url.append(result)
        return images_url


    async def get_title(self):
        """Get title of image to solve"""

        while True:
            title = await self.pictures_of()
            if title is None: # 타이틀이 안나오는 경우 다시 시도
                print("title이 None이므로 다시시도")
                time.sleep(0.3)
                continue
            else:
                break

        print("get_description_element에서 추출한 타이틀 :",title)
        group = Title.get_title_group(title)

        # 아직 존재하지 않는 키워드
        if group == None:
            print(f'{title}은 group이 존재하지 않습니다')
            if Title.is_title_korean(title):
                try:
                    title_english = Title.translate_english_to_korean(title)
                    new_title_list: list = Title.append_simillar_title(title_english)
                    new_title_list.append(title)  # 한글 타이틀도 추가
                except:
                    print('google translate api error')
                    new_title_list = []
            else:
                new_title_list = Title.append_simillar_title(title)
            print(f"존재하지 않는 레이블 -> 임시 타이틀 추가 : {new_title_list}")
            self.title = "no_title_temp"
        else:
            # 지정하는 제목은 영어 (한글 경로 읽지 못함)
            self.title = Title.get_first_english_title_in_group(group)
            # self.title = await self.search_title(title)

    async def check_detection(self, timeout):
        print("check detection")
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
