import asyncio
import time

import setting
from new_browser import show_page



if __name__ == '__main__':
     print("-------모드 선택-------")
     print("1 : OpenCV 테스트")
     print("2 : PlayWright 화면")
     print("나머지 : 실행")
     arg = input()
     proxy = "none"
     if proxy.lower() == "none":
          proxy = None

     if (arg == "1"):
          from solverecaptchas.solver import Solver
          client = Solver(setting.pageurl, setting.sitekey, proxy=proxy)
          img_path = input("이미지 경로 : ")
          image_class = input("이미지 종류 (ex : \"bus\") : ")
          result = asyncio.run(client.start_test(img_path=img_path,image_class=image_class))
          if result:
               print(result)
          exit(0)
     elif (arg == "2"):
          print("import Solver..")
          from solverecaptchas.solver import Solver
          print("load Model..")
          client = Solver(setting.pageurl, setting.sitekey, proxy=proxy)
          print("run..")
          result = asyncio.run(client.start_all_page(audio=False))
          if result:
               print(result)
               time.sleep(10)
     else: # 실행
          from solverecaptchas.solver import Solver
          client = Solver(setting.pageurl, setting.sitekey, proxy=proxy)
          result = asyncio.run(client.start())
          if result:
               print(result)

