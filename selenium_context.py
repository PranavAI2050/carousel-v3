from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from contextlib import contextmanager

@contextmanager
def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    try:
        yield driver
    finally:
        driver.quit()
