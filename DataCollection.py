from googleapiclient.discovery import build
import pandas as pd

data_api = []

# YouTube Search Categories
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

# YouTube API Client
api_key = "AIzaSyDJkCMOio3Xcu5nrKiE7IcToGRNli7IMpA"
youtube = build("youtube", "v3", developerKey=api_key)

# Searches and collects videos for a given category/search
def search_videos(cat_input, target, cat_name):
    videos = []
    next_page_token = None
    count = 0

    # Collects the target amount of videos
    while count < target:
        request = youtube.search().list(
            part="snippet",
            q=cat_input,
            type='video',
            maxResults=min(50, target - count),
            pageToken=next_page_token
        )
        response = request.execute()
        next_page_token = response.get("nextPageToken")

        # List of video ids for fetching full stats
        video_ids = []
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            if video_id:
                video_ids.append(video_id)

        # Fetch detailed stats and metadata for each video
        stats_request = youtube.videos().list(
            part="statistics,status,snippet, contentDetails",
            id=",".join(video_ids)
        )
        stats_response = stats_request.execute()

        # Extract relevant data from video
        for item in stats_response.get('items', []):
            video_id = item['id']
            title = item['snippet']['title']
            description = item['snippet']['description']
            channel_title = item['snippet']['channelTitle']
            tags = item['snippet'].get('tags')
            views = item['statistics'].get('viewCount')
            likes = item['statistics'].get('likeCount')
            comments = item['statistics'].get('commentCount')
            favorites = item['statistics'].get('favoriteCount')
            duration = item['contentDetails'].get('duration')
            upload_date = item['snippet']['publishedAt']

            # Append video info to video dictionary
            videos.append({
                "video_id": video_id,
                "title": title,
                "description": description,
                "channel_title": channel_title,
                "tags": tags,
                "views": views,
                "likes": likes,
                "comments": comments,
                "favorites": favorites,
                "duration": duration,
                "upload_date": upload_date,
                "category": cat_name
            })

        count += len(video_ids)

    # Clean up description prepare for csv
    for v in videos:
        if 'description' in v and v['description']:
            # Replace newlines and carriage returns with a space
            v['description'] = v['description'].replace('\n', ' ').replace('\r', ' ').strip()

    return videos

# Main Method
if __name__ == "__main__":

    # Loop through each category and collect 300 videos
    for yt_cat_name, yt_cat_search in yt_cat_dict.items():
        print("Collecting videos for:", yt_cat_name)
        data_api.extend(search_videos(yt_cat_search, 300, yt_cat_name))

    # Convert collected data into DataFrame
    print("Total Videos Collected:", len(data_api))

    # Save DataFrame to a CSV file
    df = pd.DataFrame(data_api)
    df.to_csv("youtube_data.csv", index=False, encoding="utf-8")
    print("Saved", len(df), "videos to file: youtube_data.csv")