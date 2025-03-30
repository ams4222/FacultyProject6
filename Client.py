from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import subprocess
import time

websites = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.reddit.com",
    "https://www.yahoo.com",
    "https://www.instagram.com",
    "https://www.linkedin.com",
    "https://www.twitter.com"
]

# SOCKS5 Proxy Configuration
SOCKS_PROXY_PORT = "" # Add proxy port when tunnel is running

chrome_options = Options()
# Set up the SOCKS5 proxy
#chrome_options.add_argument('--proxy-server=socks5://127.0.0.1:' + SOCKS_PROXY_PORT)

# Add additional arguments
chrome_options.page_load_strategy = 'normal'
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--remote-debugging-port=9222')

service = webdriver.ChromeService(log_output=subprocess.STDOUT)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    #visit websites
    for website in websites:
        driver.get(website)
        print(f"Visited: {website} - Title: {driver.title}")

    # Play YouTube video
    driver.get("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    video = driver.find_element(By.XPATH, "//*[@id=\"thumbnail\"]")
    video = driver.find_element(By.ID, "movie_player")
    video.send_keys(Keys.SPACE)
    time.sleep(2)
    video.click()
    time.sleep(2)
    print("Video played")

finally:
    driver.quit()
