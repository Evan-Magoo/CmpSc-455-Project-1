from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import math

# Selenium Setup
options = Options()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)
driver.set_window_size(1920,1080)

# YouTube Category/Searches
yt_cat_dict = {
    "Music": "music",
    "Entertainment": "entertainment",
    "Gaming": "gaming",
    "People & Blogs": "people+%26+blogs",
    "How-to & Style": "how-to+%26+style",
    "Comedy": "comedy",
    "News & Politics": "news+%26+politics",
    "Science & Technology": "science+%26+technology",
    "Education": "education",
    "Film & Animation": "film+%26+animation"
}

# Youtube Search URL
yt_url = "https://www.youtube.com/results?search_query={category}"

# Custom Header for Web Requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/127.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Scrapes YouTube search results for a specific category/search
def scrape_videos(cat_input, target, cat_name):
    # Create and open Youtube Search
    url = yt_url.format(category = cat_input)
    driver.get(url)

    scroll_pause = 3 # Time to wait between scrolls for loading
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    # Continuously scroll page until target videos are collected or page end is reached
    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(scroll_pause) # Allow time for videos to load

        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        count = len(driver.find_elements("id", "thumbnail")) # Count Videos Found

        # Progress Bar
        j = math.floor((count / target) * 20)
        bar = ""
        for i in range(0,21):
            if i <= j:
                bar = bar + "■"
            else:
                bar = bar + "-"
        print(f"\r  [{bar}] - Collected: {count} Videos", end="", flush=True)

        # Stop Conditions
        if new_height == last_height: # No more content to load
            print("\n  !Out of Scroll Height!")
            break
        elif count >= target: # Reached target video count
            print("\n  !Target Value Reached!")
            break
        last_height = new_height

    # Parse HTML Content
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    video_links = []
    videos = soup.find_all("a",id="thumbnail")
    verified = 0

    # Extract Video Links
    for video in videos:
        try:
            href = video.get("href") # Get video link from thumbnail
            if href and href.startswith("/watch"): # Keep only valid YouTube watch links
                verified += 1
                video_link = "https://www.youtube.com" + href
                entry = {
                    "link": video_link,
                    "category": cat_name
                }
                video_links.append(entry)

        except Exception as e:
            # Skip videos that fail to load
            print('  ',"Skipping video due to error:", e)
            continue

    print("  ", verified, "of these videos were verified as videos")

    # Remove duplicates
    unique_videos = {v["link"]: v for v in video_links}.values()
    print("  ", len(unique_videos), "were unique")

    return list(unique_videos)

# Main Method
if __name__ == "__main__":
    all_urls = []

    # Loop through each category and collect videos
    for yt_cat_name, yt_cat_search in yt_cat_dict.items():
        print("------------------------------------")
        print("Collecting videos for:", yt_cat_name)
        all_urls.extend(scrape_videos(yt_cat_search, 1500, yt_cat_name))

    # Remove duplicate links across categories
    all_urls = list({v["link"]: v for v in all_urls}.values())

    print(len(all_urls), "videos were scraped")

    # Save Collected Links to Text File
    with open("youtube_videos.txt","w", encoding="utf-8") as file:
        for v in all_urls:
            file.write(f"{v['link']},{v['category']}\n")

    print("All videos added to youtube_videos.txt")