pkill tor
sleep 2
while true; do
    echo "[*] Starting Tor service..."
    service tor start
    sleep 5  # Give Tor a few seconds to initialize

    echo "[*] Attempting to access check.torproject.org via Tor proxy..."
    response=$(curl -x socks5h://127.0.0.1:9050 -s --max-time 10 https://check.torproject.org)

    if echo "$response" | grep -q "Congratulations. This browser is configured to use Tor"; then
        echo "[+] Tor is working! Response received within 10 seconds."
        break
    else
        echo "[-] No valid response from Tor within 10 seconds. Killing Tor and retrying..."
        pkill tor
        sleep 2
    fi
done