import time

from playwright.sync_api import sync_playwright

def show_page():
    # Launch the browser (you can choose 'chromium', 'firefox', or 'webkit')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        # playwright = await async_playwright().start()

        browser = playwright.webkit.launch(
            headless=False,
            proxy=None
        )
        page = browser.new_page()

        # Navigate to a specific URL
        page.goto('https://www.acmicpc.net/login?next=%2F')
        time.sleep(10)
        page.frames[0].wait


        # Close the browser
        browser.close()
