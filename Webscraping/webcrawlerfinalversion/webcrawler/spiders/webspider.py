

import scrapy
from scrapy.selector import Selector
from bs4 import BeautifulSoup
from datetime import datetime
import json
import re

class WebspiderSpider(scrapy.Spider):
    name = 'webspider'
    allowed_domains = ["statutes.capitol.texas.gov"]
    chapters = [
        '1', '11', '12', '13', '14', '15', '16', '31', '32', '33', '34', '35', '36', '37', '59', '61', '62', '63', '64', '65', '66', '67', '89', '91', '92', '93', '94', '95', '96', '97', '98', '119', '121', '122', '123', '124',
        '125', '126', '149', '152', '154', '155', '156', '157', '158', '159', '160', '180', '181', '182', '183', '184', '185', '187', '199', '201', '202', '203', '204', '271',
        '273', '274', '275', '276', '278', '279', '280', '281', '301', '302', '303', '304', '305', '306', '307', '308', '339', '341', '342', '343', '344', '345', '346', '349', '350', '351', '352',
        '353', '354', '371', '391', '392', '393', '394', '395', '396', '397'
    ]
    start_urls = [f"https://statutes.capitol.texas.gov/Docs/FI/htm/FI.{chapter}.htm" for chapter in chapters]

    def __init__(self, *args, **kwargs):
        super(WebspiderSpider, self).__init__(*args, **kwargs)
        self.volunteer_id = "2132"  # Replace with your actual volunteer ID
        self.location = "Gapyeong-eup, Gapyeong-gun, South Korea"  # Replace with your actual location

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p', class_='left')

        data = []
        current_section = None
        section_texts = []

        for p in paragraphs:
            text = p.get_text(strip=True)
            chapter_match = re.match(r'Sec\.\s*(\d+)\.', text)
            if chapter_match:
                chapter = chapter_match.group(1)
                if current_section:
                    data.append({
                        "text": " ".join(section_texts).strip(),
                        "metadata": {
                            "date_downloaded": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                            "site_url": response.url,
                            "extra_data": {
                                "chapter": chapter,
                                "section": current_section
                            }
                        },
                        "volunteer_id": self.volunteer_id,
                        "location": self.location
                    })
                current_section = text.split(' ')[1].strip('.')
                section_texts = [text]
            else:
                section_texts.append(text)

        if current_section:
            data.append({
                "text": " ".join(section_texts).strip(),
                "metadata": {
                    "date_downloaded": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                    "site_url": response.url,
                    "extra_data": {
                        "chapter": chapter,
                        "section": current_section
                    }
                },
                "volunteer_id": self.volunteer_id,
                "location": self.location
            })

        # Append to the JSON file instead of overwriting it
        with open('finance_commission.json', 'a', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

# This code allows to go to the link and use selenium to open the tab for each chapter. This did not work, and I was not able to debug it.
"""
import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from bs4 import BeautifulSoup
from datetime import datetime
import json
import logging

class WebspiderSpider(scrapy.Spider):
    name = "webspider"
    start_urls = ['https://statutes.capitol.texas.gov/Index.aspx']

    def __init__(self, *args, **kwargs):
        super(WebspiderSpider, self).__init__(*args, **kwargs)
        service = ChromeService(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service)
        self.driver.implicitly_wait(10)
        logging.basicConfig(level=logging.INFO)
        
        self.volunteer_id = "2132"  # Replace with your actual volunteer ID
        self.location = "Gapyeong-eup, Gapyeong-gun, South Korea"  # Replace with your actual location

    def start_requests(self):
        for url in self.start_urls:
            logging.info(f"Starting request for URL: {url}")
            yield scrapy.Request(url=url, callback=self.parse, dont_filter=True)

    def parse(self, response):
        self.driver.get(response.url)
        logging.info("Page loaded: %s", response.url)

        try:
            select_code = Select(WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, 'ctl00_ContentPlaceHolder1_QSearch_cboQuickCode'))
            ))
            select_code.select_by_visible_text('Finance Code')
            logging.info("Selected 'Finance Code'")

            select_chapter = Select(WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'ctl00_ContentPlaceHolder1_QSearch_cboQuickChapter'))
            ))

            for option in select_chapter.options[1:]:  # Skipping the first option as it might be placeholder
                chapter_text = option.text
                select_chapter.select_by_visible_text(chapter_text)
                logging.info("Selected chapter: %s", chapter_text)

                go_button = self.driver.find_element(By.ID, 'ctl00_ContentPlaceHolder1_QSearch_btnQSGo')
                ActionChains(self.driver).click(go_button).perform()
                logging.info("Clicked 'Go' button")

                WebDriverWait(self.driver, 60).until(EC.new_window_is_opened(self.driver.window_handles))
                new_tab_handle = self.driver.window_handles[-1]
                self.driver.switch_to.window(new_tab_handle)
                logging.info("Switched to new tab: %s", self.driver.current_url)

                # Wait until all p.left elements are loaded
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'p.left'))
                )
                logging.info("All p.left elements are loaded")

                response = HtmlResponse(url=self.driver.current_url, body=self.driver.page_source, encoding='utf-8')
                logging.info("Page source length: %d", len(response.text))
                self.extract_and_save_text(response, chapter_text)

                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])

        except Exception as e:
            logging.error("An error occurred: %s", str(e))

    def extract_and_save_text(self, response, chapter):
        soup = BeautifulSoup(response.body, 'html.parser')
        paragraphs = soup.find_all('p', class_='left')

        data = []
        current_section = None
        section_texts = []

        for p in paragraphs:
            text = p.get_text(strip=True)
            if text.startswith('Sec.'):
                if current_section:
                    data.append({
                        "text": " ".join(section_texts).strip(),
                        "metadata": {
                            "date_downloaded": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                            "site_url": response.url,
                            "extra_data": {
                                "chapter": chapter,
                                "section": current_section
                            }
                        },
                        "volunteer_id": self.volunteer_id,
                        "location": self.location
                    })
                current_section = text.split(' ')[1].strip('.')
                section_texts = [text]
            else:
                section_texts.append(text)

        if current_section:
            data.append({
                "text": " ".join(section_texts).strip(),
                "metadata": {
                    "date_downloaded": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                    "site_url": response.url,
                    "extra_data": {
                        "chapter": chapter,
                        "section": current_section
                    }
                },
                "volunteer_id": self.volunteer_id,
                "location": self.location
            })

        with open('finance_commission.json', 'a', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

    def closed(self, reason):
        self.driver.quit()
        logging.info("Spider closed: %s", reason)

"""
