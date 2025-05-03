from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import *
import subprocess
import signal
import csv
import random


websites = []
with open('/app/domains.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        websites.append("https://" + row['Domain'])    

# SOCKS5 Proxy Configuration
SOCKS_PROXY_PORT = "9050" 

chrome_options = Options()
# Set up the SOCKS5 proxy
chrome_options.add_argument('--proxy-server=socks5://127.0.0.1:' + SOCKS_PROXY_PORT)

# Add additional arguments
chrome_options.page_load_strategy = 'normal'
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--edge-skip-compat-layer-relaunch')
chrome_options.add_argument('--remote-debugging-port=9222')

service = webdriver.ChromeService(log_output=subprocess.STDOUT)
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.set_page_load_timeout(90)

index = 0
for website in websites:
    # Visit websites
    try:
        # Starts tcpdump capture
        tcpdump = subprocess.Popen(["tcpdump", "-i", "any", "-w", "/app/pcap/traffic-" + str(index) + ".pcap"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Visits website and prints website
        driver.get(website)
        print(f"Visited: {website} - Title: {driver.title}", flush=True)
        
        # Ends tcpdump capture
        tcpdump.send_signal(signal.SIGINT)
        tcpdump.wait()
        
        # Increases traffic file index
        index += 1

    # Catches any webdriver or connection exceptions
    except WebDriverException as e:
        print("Exception occurred: ", e)
        #if "Message: unknown error: net::ERR_SOCKS_CONNECTION_FAILED" in str(e):
            #subprocess.run(["bash", "/app/restart_tor.sh"])
    except Exception as e:
        print("Exception occured: ", e)

# Shutdown webdriver
driver.quit()

# Shutdown OpenVPN
subprocess.Popen(['echo', 'signal', 'SIGTERM', '|nc', '127.0.0.1', '999'])
