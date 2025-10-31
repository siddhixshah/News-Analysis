# Streamlit app: NSE small-cap news fetcher using GNews (single-file)
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
- If you prefer another news provider (NewsAPI, SerpApi, Webz.io, etc.), replace the fetch_news() implementation accordingly.
- You can supply a CSV file of tickers with two columns: ticker,company_name. Or use the small built-in list to start.
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import requests
import streamlit as st
from dateutil import parser

# Optional sentiment (lightweight heuristic). If you want more accurate sentiment, install vaderSentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    VADER_AVAILABLE = False
    # fallback lexicons (very small and simple)
    POS_WORDS = set(["good","up","gain","positive","beat","beats","beat","outperform","buy","profit","growth","record","surge","rise","benefit","upgrade"]) 
    NEG_WORDS = set(["loss","down","drop","decline","miss","missed","warn","downgrade","sell","fall","fraud","fine","slump","recall","delay","cut"]) 


# ------------------------
# Helper utilities
# ------------------------

def simple_sentiment(text: str) -> str:
    """Return 'positive'/'neutral'/'negative' using VADER if available, else a tiny lexicon heuristic."""
    if not text:
        return "neutral"
    if VADER_AVAILABLE:
        vs = analyzer.polarity_scores(text)
        comp = vs["compound"]
        if comp >= 0.05:
            return "positive"
        elif comp <= -0.05:
            return "negative"
        else:
            return "neutral"
    # tiny heuristic
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos - neg >= 2:
        return "positive"
    if neg - pos >= 2:
        return "negative"
    return "neutral"


# ------------------------
# News fetching (GNews)
# ------------------------

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"

@st.cache_data(ttl=60*5)
def cached_fetch_news(query: str, from_iso: str, to_iso: str, max_pages: int = 3, page_size: int = 100) -> List[Dict]:
    """Fetch news using GNews API with simple rate-limit/backoff handling.

    - Expects environment variable GOOGLE_NEWS_API_KEY to be set.
    - Returns a list of article dicts with keys: title, description, content, url, source, publishedAt
    """
    api_key = os.getenv("GOOGLE_NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GOOGLE_NEWS_API_KEY not set. See app comments for how to set it.")

    headers = {"Accept": "application/json"}
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
            "in": "title,description"
        }
        # exponential backoff retries
        retries = 0
        while True:
            try:
                resp = requests.get(GNEWS_SEARCH_URL, params=params, headers=headers, timeout=15)
            except requests.RequestException as e:
                retries += 1
                if retries > 3:
                    st.warning(f"Network error while fetching news: {e}")
                    return all_articles
                sleep_for = 2 ** retries
                time.sleep(sleep_for)
                continue

            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("articles", [])
                if not articles:
                    return all_articles
                for a in articles:
                    # normalize fields
                    article = {
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "content": a.get("content"),
                        "url": a.get("url"),
                        "source": a.get("source", {}).get("name"),
                        "publishedAt": a.get("publishedAt")
                    }
                    all_articles.append(article)
                # if fewer than page_size returned, we're done
                if len(articles) < page_size:
                    return all_articles
                # else continue to next page
                break

            # handle rate limits or server errors
            if resp.status_code in (429, 503, 500):
                retries += 1
                if retries > 4:
                    st.warning(f"News API returned status {resp.status_code}; giving up.")
                    return all_articles
                sleep_for = 2 ** retries
                time.sleep(sleep_for)
                continue
            else:
                # other errors - show and stop
                try:
                    st.error(f"News API error {resp.status_code}: {resp.text}")
                except Exception:
                    st.error(f"News API error {resp.status_code}")
                return all_articles
    return all_articles


# ------------------------
# UI and app logic
# ------------------------

st.set_page_config(page_title="NSE Small-cap News Explorer", layout="wide")
st.title("NSE Small-cap — Google News Explorer")

st.markdown(
    "Use this app to pick a small-cap NSE stock and a date range, then fetch related news using the (GNews) Google News API.\n\n"
    "Make sure you set the environment variable `GOOGLE_NEWS_API_KEY` before running."
)

# Provide a tiny built-in sample list (ticker, company)
BUILTIN_SMALLCAPS = [
    ("PNB", "Punjab National Bank"),
    ("IDFCFIRSTB", "IDFC First Bank"),
    ("TVSMOTOR", "TVS Motor Company"),
    ("MSTCLTD", "MSTC Limited"),
    ("TATACHEM", "Tata Chemicals"),
]

st.sidebar.header("Tickers")
upload = st.sidebar.file_uploader("Upload CSV of tickers (ticker,company_name) — optional", type=["csv"])

if upload is not None:
    try:
        df_tickers = pd.read_csv(upload)
        if df_tickers.shape[1] >= 2:
            options = list(df_tickers.iloc[:, 0].astype(str) + " | " + df_tickers.iloc[:, 1].astype(str))
            ticker_map = {opt.split(" | ")[0]: opt.split(" | ")[1] for opt in options}
        else:
            st.sidebar.warning("CSV must have at least two columns: ticker, company_name. Using built-in list.")
            ticker_map = {t: c for t, c in BUILTIN_SMALLCAPS}
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        ticker_map = {t: c for t, c in BUILTIN_SMALLCAPS}
else:
    ticker_map = {t: c for t, c in BUILTIN_SMALLCAPS}

# Optional: explain how to auto-fetch small-cap tickers
with st.sidebar.expander("How to supply tickers / auto-fetch idea (instructions)"):
    st.write(
        "You can upload a CSV with columns: ticker,company_name.\nIf you want to auto-fetch a list of small-cap tickers programmatically, you could use an NSE scraping library or an official data provider and then feed that CSV here."
    )

selected_ticker = st.sidebar.selectbox("Select ticker", options=list(ticker_map.keys()))
company_name = ticker_map[selected_ticker]

# Date range options
range_label = st.sidebar.selectbox("Date range", ["Last 1 week", "Last 1 month", "Last 3 months", "Last 6 months"])
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
from_iso = from_dt.isoformat() + "Z"
to_iso = to_dt.isoformat() + "Z"

st.sidebar.markdown(f"**Query period:** {from_dt.date().isoformat()} — {to_dt.date().isoformat()}")

# Sentiment toggle
use_sentiment = st.sidebar.checkbox("Calculate sentiment (lightweight)", value=False)

# Search settings
max_pages = st.sidebar.slider("Max pages to fetch (per query)", min_value=1, max_value=5, value=2)
page_size = st.sidebar.selectbox("Results per page (max depends on API plan)", [10, 20, 50, 100], index=1)

# Build query: try ticker and company name
query = f'"{selected_ticker}" OR "{company_name}"'

if st.sidebar.button("Fetch news"):
    with st.spinner("Fetching news — this may take a few seconds..."):
        try:
            articles = cached_fetch_news(query=query, from_iso=from_iso, to_iso=to_iso, max_pages=max_pages, page_size=page_size)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        if not articles:
            st.info("No articles found for this query and date range.")
        else:
            # Normalize into DataFrame
            df = pd.DataFrame(articles)
            # parse dates
            def parse_date(x):
                try:
                    return parser.isoparse(x)
                except Exception:
                    try:
                        return parser.parse(x)
                    except Exception:
                        return pd.NaT
            df['publishedAt_parsed'] = df['publishedAt'].apply(parse_date)
            df = df.sort_values('publishedAt_parsed', ascending=False)

            if use_sentiment:
                df['sentiment'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
                df['sentiment'] = df['sentiment'].apply(simple_sentiment)

            # Display summary
            st.success(f"Fetched {len(df)} articles for {selected_ticker} ({company_name})")
            col1, col2 = st.columns([2, 1])
            with col2:
                if st.button("Download CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", data=csv, file_name=f"{selected_ticker}_news_{from_dt.date().isoformat()}_{to_dt.date().isoformat()}.csv")

            # Interactive list
            for idx, row in df.iterrows():
                pub = row.get('publishedAt_parsed')
                pub_str = pub.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(pub) else row.get('publishedAt')
                title = row.get('title') or "(no title)"
                desc = row.get('description') or ""
                source = row.get('source') or "Unknown"
                url = row.get('url') or ""
                sentiment = row.get('sentiment') if 'sentiment' in row else None

                st.markdown(f"**{title}**  ")
                st.markdown(f"*{source} — {pub_str}*  ")
                if desc:
                    st.markdown(desc)
                if url:
                    st.markdown(f"[Read full article]({url})")
                if sentiment is not None:
                    st.markdown(f"**Sentiment:** {sentiment}")
                st.divider()

else:
    st.info("Pick a ticker and date range from the sidebar, then click 'Fetch news'.")

# Footer and instructions
st.markdown("---")
st.caption(
    "App built for prototyping. To change providers or increase rate-limits, swap the fetch_news implementation to your preferred News API and set its API key in the environment."
)
