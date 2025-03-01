from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.keys import Keys
import subprocess
import time

# SOCKS5 Proxy Configuration
SOCKS_PROXY_HOST = "127.0.0.1"
SOCKS_PROXY_PORT = "" # Add proxy port when tunnel is running

# List of popular websites to visit
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

# Set up Firefox options
options = Options()
options.add_argument("--headless")  # Run headless
options.binary_location = "/usr/bin/firefox"  # Sets the location for the browser executable
options.set_preference("gfx.webrender.all", False)
options.set_preference("layers.acceleration.disabled", True)

# Set SOCKS5 proxy preferences
#options.set_preference("network.proxy.type", 1)  # Manual proxy configuration
#options.set_preference("network.proxy.socks", SOCKS_PROXY_HOST)
#options.set_preference("network.proxy.socks_port", int(SOCKS_PROXY_PORT))
#options.set_preference("network.proxy.socks_version", 5)  # Set SOCKS5
#options.set_preference("network.proxy.socks_remote_dns", True)  # Resolve DNS through proxy

# Initialize WebDriver
#service = Service(GeckoDriverManager().install(), log_output=subprocess.STDOUT) # UNCOMMENT TO OUTPUT LOGS TO TERMINAL
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service, options=options)

try:
    for website in websites:
        driver.get(website)
        print(f"Visited: {website} - Title: {driver.title}")

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
