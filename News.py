# Streamlit app: NSE small-cap news fetcher using GNews (auto-fetch version)
# Save this file as streamlit_nse_smallcap_news.py and run: streamlit run streamlit_nse_smallcap_news.py
# Requirements: pip install streamlit requests pandas python-dateutil
# Optional (for better sentiment): pip install vaderSentiment

"""
How to set your API key (Linux / macOS):
  export GOOGLE_NEWS_API_KEY="your_gnews_api_key_here"
On Windows (PowerShell):
  setx GOOGLE_NEWS_API_KEY "your_gnews_api_key_here"

Notes:
- This app uses the GNews API (https://gnews.io). The environment variable name is GOOGLE_NEWS_API_KEY.
- Automatically fetches news on selection change (no button needed).
- You can supply a CSV file of tickers with two columns: ticker,company_name. Or use the small built-in list.
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import requests
import streamlit as st
from dateutil import parser

# Optional sentiment (lightweight heuristic)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False
    POS_WORDS = set(["good","up","gain","positive","beat","outperform","buy","profit","growth","record","surge","rise","benefit","upgrade"])
    NEG_WORDS = set(["loss","down","drop","decline","miss","warn","sell","fall","slump","delay","cut"])

def simple_sentiment(text: str) -> str:
    if not text:
        return "neutral"
    if VADER_AVAILABLE:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
    text = text.lower()
    pos = sum(w in text for w in POS_WORDS)
    neg = sum(w in text for w in NEG_WORDS)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    else:
        return "neutral"

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"

@st.cache_data(ttl=60*5)
def cached_fetch_news(query: str, from_iso: str, to_iso: str, max_pages: int = 3, page_size: int = 50) -> List[Dict]:
    api_key = os.getenv("GOOGLE_NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GOOGLE_NEWS_API_KEY not set.")

    all_articles = []
    for page in range(1, max_pages + 1):
        params = {
            "apikey": api_key,
            "q": query,
            "from": from_iso,
            "to": to_iso,
            "lang": "en",
            "max": page_size,
            "page": page,
        }
        retries = 0
        while True:
            try:
                resp = requests.get(GNEWS_SEARCH_URL, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get("articles", [])
                    if not articles:
                        return all_articles
                    for a in articles:
                        all_articles.append({
                            "title": a.get("title"),
                            "description": a.get("description"),
                            "content": a.get("content"),
                            "url": a.get("url"),
                            "source": a.get("source", {}).get("name"),
                            "publishedAt": a.get("publishedAt"),
                        })
                    if len(articles) < page_size:
                        return all_articles
                    break
                elif resp.status_code in (429, 500, 503):
                    retries += 1
                    if retries > 3:
                        return all_articles
                    time.sleep(2 ** retries)
                    continue
                else:
                    return all_articles
            except Exception:
                retries += 1
                if retries > 3:
                    return all_articles
                time.sleep(2 ** retries)
    return all_articles

st.set_page_config(page_title="NSE Small-cap News Explorer", layout="wide")
st.title("📈 NSE Small-cap — Google News Explorer (Auto-Fetch)")

st.markdown("Automatically fetches the latest news for selected small-cap NSE stocks using Google News API.")

BUILTIN_SMALLCAPS = [
    ("PNB", "Punjab National Bank"),
    ("IDFCFIRSTB", "IDFC First Bank"),
    ("TVSMOTOR", "TVS Motor Company"),
    ("MSTCLTD", "MSTC Limited"),
    ("TATACHEM", "Tata Chemicals"),
]

st.sidebar.header("Ticker Selection")
upload = st.sidebar.file_uploader("Upload CSV (ticker,company_name) — optional", type=["csv"])
if upload is not None:
    try:
        df = pd.read_csv(upload)
        ticker_map = dict(zip(df.iloc[:,0], df.iloc[:,1]))
    except Exception as e:
        st.sidebar.warning(f"Error reading CSV: {e}. Using built-in list.")
        ticker_map = dict(BUILTIN_SMALLCAPS)
else:
    ticker_map = dict(BUILTIN_SMALLCAPS)

selected_ticker = st.sidebar.selectbox("Select stock ticker", list(ticker_map.keys()))
company_name = ticker_map[selected_ticker]

range_label = st.sidebar.selectbox("Select date range", ["Last 1 week","Last 1 month","Last 3 months","Last 6 months"])
now = datetime.utcnow()
if range_label == "Last 1 week":
    from_dt = now - timedelta(weeks=1)
elif range_label == "Last 1 month":
    from_dt = now - timedelta(days=30)
elif range_label == "Last 3 months":
    from_dt = now - timedelta(days=90)
else:
    from_dt = now - timedelta(days=180)

to_dt = now
from_iso, to_iso = from_dt.isoformat()+"Z", to_dt.isoformat()+"Z"

use_sentiment = st.sidebar.checkbox("Show sentiment", value=True)
max_pages = st.sidebar.slider("Pages",1,5,2)
page_size = st.sidebar.selectbox("Articles per page", [10,20,50,100], index=2)

query = f'"{selected_ticker}" OR "{company_name}"'

st.subheader(f"News for {selected_ticker} — {company_name}")

try:
    with st.spinner("Fetching news..."):
        articles = cached_fetch_news(query, from_iso, to_iso, max_pages, page_size)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if not articles:
    st.info("No news articles found for this stock and range.")
else:
    df = pd.DataFrame(articles)
    def parse_date(x):
        try:
            return parser.isoparse(x)
        except Exception:
            return pd.NaT
    df['publishedAt_parsed'] = df['publishedAt'].apply(parse_date)
    df = df.sort_values('publishedAt_parsed', ascending=False)

    if use_sentiment:
        df['sentiment'] = (df['title'].fillna('') + '. ' + df['description'].fillna('')).apply(simple_sentiment)

    st.success(f"Fetched {len(df)} articles for {selected_ticker} ({company_name})")

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, file_name=f"{selected_ticker}_news.csv")

    for _, row in df.iterrows():
        title = row.get('title') or "(No title)"
        desc = row.get('description') or ""
        src = row.get('source') or "Unknown"
        date = row.get('publishedAt_parsed')
        date_str = date.strftime('%Y-%m-%d %H:%M UTC') if pd.notna(date) else ''
        link = row.get('url') or ""
        st.markdown(f"**{title}**")
        st.caption(f"{src} — {date_str}")
        if desc:
            st.write(desc)
        if link:
            st.markdown(f"[Read more]({link})")
        if use_sentiment:
            st.markdown(f"**Sentiment:** {row.get('sentiment','neutral')}\n")
        st.divider()

st.markdown("---")
st.caption("App auto-fetches GNews results for small-cap NSE stocks based on selected range.")
