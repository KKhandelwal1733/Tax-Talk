import requests

url = "https://cleartax.in/s/gst-news-and-announcements"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise error for bad status codes
    
    with open("page.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Download successful!")
except requests.RequestException as e:
    print(f"Error: {e}")