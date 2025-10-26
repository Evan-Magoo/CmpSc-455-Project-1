import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import re
import math



data_scrape = []

# Selenium Setup
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--disable-extensions")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--blink-settings=imagesEnabled=false")  # disables image loading
options.add_argument("--mute-audio")
options.add_argument("--disable-background-networking")
options.add_argument("--disable-software-rasterizer")
options.add_argument("--disable-renderer-backgrounding")
driver = webdriver.Chrome(options=options)
driver.set_window_size(1920,1080)

# Custom Header for Web Requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/127.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# Main Scraper
def scrape_stats(video_url, cat):
    driver.get(video_url)
    driver.execute_script("window.scrollBy(0, 2000);")
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract Video ID
    # noinspection PyBroadException
    try:
        match = re.search(r'"target":\{"videoId":"([^"]+)"\}', str(soup))
        if match:
            video_id = match
        else:
            video_id = ""
    except Exception:
        video_id = video_url

    # Extract Title
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.text.strip()
        if title.endswith(" - YouTube"):
            title = title[:-10]
    else:
        title = "Unknown"

    # Extract Description
    description = ""
    description_html = soup.find("span", class_="yt-core-attributed-string yt-core-attributed-string--white-space-pre-wrap", dir="auto")
    if description_html:
        for span in description_html.find_all(
                "span", class_="yt-core-attributed-string--link-inherit-color"
        ):
            description += span.text
    else:
        description = ""  # or "No description available"

    # Extract Channel Title
    channel_title = ""
    channel_title_html = soup.find("div", class_="style-scope ytd-channel-name")
    if channel_title_html:
        link_tag = channel_title_html.find("a", class_="yt-simple-endpoint style-scope yt-formatted-string")
        if link_tag and link_tag.text:
            channel_title = link_tag.text.strip()
    else:
        # Fallback: try JSON-based extraction
        match = re.search(r'"channel":\{"simpleText":"([^"]+)"\}', str(soup))
        if match:
            channel_title = match.group(1)
        else:
            channel_title = "Unknown"

    # Extract Tags
    tags = []
    tags_html = soup.find_all("meta", property="og:video:tag")
    for tag_html in tags_html:
        if tag_html.has_attr("content"):
            tags.append(tag_html["content"])
    if not tags:
        tags = "None"

    # Extract Likes
    likes_html = soup.find("div", class_="ytSegmentedLikeDislikeButtonViewModelSegmentedButtonsWrapper")
    if likes_html:
        like_text_div = likes_html.find("div", class_="yt-spec-button-shape-next__button-text-content")
        if like_text_div:
            likes_raw = like_text_div.text.strip()
        else:
            likes_raw = "0"
    else:
        match = re.search(r'"label":"([\d,.]+)\s*likes"', str(soup))
        if match:
            likes_raw = match.group(1)
        else:
            likes_raw = "0"

    # Extract Views
    views = "0"
    views_html = soup.find("span", class_="view-count style-scope ytd-video-view-count-renderer")
    if views_html:
        views_text = views_html.text.strip()
        # Example: "1,234,567 views"
        views = re.sub(r"[^\d]", "", views_text)  # remove commas and text
    else:
        # Fallback: look for a JSON pattern
        match = re.search(r'"viewCount":"(\d+)"', str(soup))
        if match:
            views = match.group(1)
        else:
            # Rare case: another JSON pattern
            match = re.search(r'"shortViewCountText":\{"simpleText":"([^"]+)"\}', str(soup))
            if match:
                raw = match.group(1).replace(",", "")
                if "K" in raw:
                    views = str(int(float(raw.replace("K", "")) * 1000))
                elif "M" in raw:
                    views = str(int(float(raw.replace("M", "")) * 1_000_000))
                elif "B" in raw:
                    views = str(int(float(raw.replace("B", "")) * 1_000_000_000))
                else:
                    views = re.sub(r"[^\d]", "", raw)
    try:
        views = int(views)
    except ValueError:
        views = 0

    # Convert likes_raw string to integer
    likes_raw = likes_raw.replace(",", "")
    if 'K' in likes_raw:
        likes = int(float(likes_raw.replace('K', '')) * 1000)
    elif 'M' in likes_raw:
        likes = int(float(likes_raw.replace('M', '')) * 1_000_000)
    elif 'B' in likes_raw:
        likes = int(float(likes_raw.replace('B', '')) * 1_000_000_000)
    else:
        try:
            likes = int(likes_raw)
        except ValueError:
            likes = 0

    # Extract Duration
    driver.execute_script("window.scrollBy(0, 500);")
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    duration_elem = soup.find("span", class_="ytp-time-duration")
    if duration_elem and duration_elem.text:
        duration = duration_elem.text.strip()
    else:
        match = re.search(r'"lengthText":\{"simpleText":"([^"]+)"\}', str(soup))
        duration = match.group(1) if match else "Unknown"

    # Extract Upload Date
    # noinspection PyBroadException
    try:
        upload_date = soup.find_all("div",class_="ytwFactoidRendererFactoid", role="text")[2].get('aria-label').replace(",","")
    except Exception:
        # noinspection PyBroadException
        try:
            match = re.search(r'"publishDate":\{"simpleText":"([^"]+)"\}', str(soup))
            if match:
                upload_date = match.group(1)
            else:
                upload_date = "Unknown"
        except Exception:
            upload_date = "Unknown"
    upload_date = upload_date.replace("Premiered ", "")
    upload_date = upload_date.replace("Streamed live on ", "")
    upload_date = upload_date.replace("Published on ", "")

    # Combine Video Data
    video = {
        "video_id": video_id,
        "title": title,
        "description": description,
        "channel_title": channel_title,
        "tags": tags,
        "views": views,
        "likes": likes,
        "duration": duration,
        "upload_date": upload_date,
        "category": cat
    }

    return video

# Main Method
if __name__ == "__main__":
    # Load video URL and corresponding category
    df = pd.read_csv('youtube_videos.txt', header=None, names=["link","category"])
    total_videos = len(df)
    print(total_videos)
    count = 0
    videos = []

    # Loop through each video and scrape stats
    for _, row in df.iterrows():
        videos.append(scrape_stats(row["link"], row["category"]))
        count += 1

        # Create Progress Bar
        j = math.floor((count / total_videos) * 20)
        bar = ""
        for i in range(0, 21):
            if i <= j:
                bar = bar + "■"
            else:
                bar = bar + "-"
        print(f"\r  [{bar}] - Converted: {count}/{total_videos} Videos", end="", flush=True)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(videos)
    df.to_csv("youtube_scrape_data.csv", index=False, encoding="utf-8")
    print("Saved", len(df), "videos to file: youtube_scrape_data.csv")