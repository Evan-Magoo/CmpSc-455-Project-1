import pandas as pd
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import emoji

data_scrape = []
data_api = []

# ----------- Fixing Scrape Data for Comparison ------------------------------------------------------------------------
def fix_scrape(df_scrape):
    df_scrape.dropna(inplace=True)

    # Extract Video ID
    for i, v in df_scrape['video_id'].items():
        if isinstance(v, str):
            match = re.search(r'"videoId":"([^"]+)"', v)
            if match:
                video_id = match.group(1)
            else:
                video_id = ""
        else:
            video_id = ""
        df_scrape.at[i, "video_id"] = video_id

    # Convert Duration into Total Seconds
    for i, d in df_scrape['duration'].items():
        try:
            parts = list(map(int, d.split(":")))
            if len(parts) == 3:
                hours, minutes, seconds = parts
            elif len(parts) == 2:
                hours = 0
                minutes, seconds = parts
            else:
                hours, minutes, seconds = 0, 0, 0
            total_time = (hours * 3600) + (minutes * 60) + seconds
        except:
            total_time = 0
        df_scrape.at[i, 'duration'] = total_time

    # Convert Upload Dates to Datetime
    for i, d in df_scrape['upload_date'].items():
        if isinstance(d, str):
            d = d.strip()
            try:
                date = datetime.strptime(d, "%b %d, %Y")
            except ValueError:
                try:
                    date = datetime.strptime(d, "%b %d %Y")
                except ValueError:
                    date = ""
        else:
            date = ""
        df_scrape.at[i, "upload_date"] = date

    return df_scrape

# ----------- Fixing API Data for Comparison ---------------------------------------------------------------------------
def fix_api(df_api):
    df_api.dropna(inplace=True)
    df_api.drop('comments', axis=1, inplace=True)
    df_api.drop('favorites', axis=1, inplace=True)

    # Convert YouTube Duration Format to Seconds
    for i, d in df_api['duration'].items():
        if isinstance(d, str):
            if 'H' in d:
                hours = int(re.search(r'(\d+)H', d).group(1))
            else:
                hours = 0

            if 'M' in d:
                minutes = int(re.search(r'(\d+)M', d).group(1))
            else:
                minutes = 0

            if 'S' in d:
                seconds = int(re.search(r'(\d+)S', d).group(1))
            else:
                seconds = 0

            total_time = (hours * 3600) + (minutes * 60) + seconds
        else:
            total_time = 0

        df_api.at[i, 'duration'] = total_time

    return df_api

# ----------- Preprocessing --------------------------------------------------------------------------------------------
def preprocessing(df, file_name):
    df = df.copy()
    df.dropna()

    # ---- Text Cleaner ----
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = text.strip()
        return text

    # ---- Likes/Views ----
    df = df.copy()
    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
    df = df[df['views'] >= 100].copy()
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)

    # ---- Title ----
    df['has_questions'] = df['title'].str.contains(r'\?', regex=True).astype(int)
    df['has_exclamations'] = df['title'].str.contains(r'\!', regex=True).astype(int)
    df['has_numbers'] = df['title'].str.contains(r'\d', regex=True).astype(int)
    df['has_emoji'] = df['title'].apply(lambda x: int(bool(emoji.emoji_list(str(x)))))
    df['title_caps_ratio'] = df['title'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()) / (len(str(x).split()) + 1e-5))
    df['emoji_count'] = df.apply(lambda r: len(emoji.emoji_list(str(r['title']) + str(r['description']))), axis=1)
    df['title_clean'] = df['title'].apply(clean_text)
    df['title_word_count'] = df['title_clean'].apply(lambda x: len(x.split()))
    df['title_char_count'] = df['title_clean'].apply(len)

    # ---- Clickbait Features ----
    clickbait_words = [
        "best", "worst", "amazing", "incredible", "crazy", "shocking", "unbelievable", "epic",
        "top", "10", "5", "7", "ranked", "list", "number", "count",
        "now", "today", "before", "never", "last chance",
        "secret", "revealed", "hidden", "you won't believe", "what happens next",
        "love", "hate", "funniest", "sad", "terrifying",
        "watch", "look", "don't miss", "check this",
        "fails", "fails compilation", "tricks", "life-changing", "extreme", "craziest"
    ]
    pattern = re.compile(r'\b(' + '|'.join(re.escape(w) for w in clickbait_words) + r')\b', re.IGNORECASE)
    df['clickbait_matches'] = df['title'].astype(str).apply(lambda x: pattern.findall(x))
    df['clickbait_count'] = df['clickbait_matches'].apply(len)
    df['clickbait_ratio'] = df['clickbait_count'] / (df['title_word_count'] + 1e-5)
    df['has_clickbait'] = (df['clickbait_count'] > 0).astype(int)
    df.drop(columns=['clickbait_matches'], inplace=True)

    # ---- Description ----
    df['des_caps_ratio'] = df['description'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()) / (len(str(x).split()) + 1e-5))
    df['des_clean'] = df['description'].apply(clean_text)
    df['des_word_count'] = df['des_clean'].apply(lambda x: len(x.split()))
    df['des_char_count'] = df['des_clean'].apply(len)
    df['des_avg_word_len'] = df['des_clean'].apply(lambda x: sum(len(w) for w in x.split()) / (len(x.split()) + 1e-5))

    # ---- Channel Title ----
    df['channel_clean'] = df['channel_title'].apply(clean_text)
    df['channel_word_count'] = df['channel_clean'].apply(lambda x: len(x.split()))
    df['channel_char_count'] = df['channel_clean'].apply(len)
    channel_counts = df['channel_title'].value_counts()
    df['channel_freq'] = df['channel_title'].map(channel_counts)

    # ---- Tags ----
    def split_tags(x):
        if isinstance(x, str):
            # split by comma, strip whitespace, remove empty strings
            return [t.strip() for t in x.split(',') if t.strip()]
        return []
    df['tags'] = df['tags'].apply(split_tags)
    df['tags_count'] = df['tags'].apply(len)

    # ---- Duration ----
    df = df[(df['duration'] > 0) & (df['duration'] <= 21600)].copy()
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

    # ---- Upload Date ----
    collection_date = datetime(2025, 10, 22) # Oct 22, 2025
    df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['upload_date'])
    df['days_since_upload'] = (collection_date - df['upload_date']).dt.days.fillna(0)
    df['upload_month'] = df['upload_date'].dt.month
    df['upload_day'] = df['upload_date'].dt.day
    df['upload_hour'] = df['upload_date'].dt.hour
    df['upload_weekday'] = df['upload_date'].dt.weekday
    df['is_weekend'] = df['upload_weekday'].isin([4,6]).astype(int)
    df = pd.get_dummies(df, columns=['upload_weekday'], prefix='weekday')
    dummy_cols = [c for c in df.columns if c.startswith('weekday_')]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # ---- Category ----
    le = LabelEncoder()
    df['category_label'] = df['category']
    df['category'] = le.fit_transform(df['category'].fillna('Unknown'))

    df = df.drop(columns=['title','title_clean','description', 'des_clean', 'channel_title', 'channel_clean', 'tags','upload_date'])

    df.to_csv(file_name, index=False, encoding="utf-8")
    print("Processed:", file_name)

# ----------- Main -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load Unprocessed DataSets
    scrape_csv = pd.read_csv('youtube_scrape_data.csv')
    api_csv = pd.read_csv('youtube_data.csv')

    # Apply Dataset-Specific Cleaning
    data_scrape = fix_scrape(scrape_csv)
    data_api = fix_api(api_csv)

    print(data_scrape)
    print(data_api)

    # Find Videos in Both
    match_ids = data_scrape['video_id'].isin(data_api['video_id'])
    data_match = data_scrape[match_ids]

    # Preprocess DataSets
    preprocessing(data_scrape, "scrape_data_processed.csv")
    preprocessing(data_api, "api_data_processed.csv")
    preprocessing(data_match, "match_data_processed.csv")
