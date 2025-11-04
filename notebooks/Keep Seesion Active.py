import time
import requests

URL = "https://after-dafter.onrender.com"
INTERVAL_MINUTES = 14

def ping_site():
    try:
        r = requests.get(URL, timeout=10)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {URL} → {r.status_code}")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")
while True:
    ping_site()
    user_input = input("type 'stop' to exit: ").strip().lower()
    if user_input == "stop":
        print(" Script stopped by user.")
        break

    #  next ping
    print(f"Waiting {INTERVAL_MINUTES} minutes for next ping...\n")
    time.sleep(INTERVAL_MINUTES * 13)
