"""
APIFY Twitter/X scraper tool.
Fetches tweets mentioning given stock tickers from the last 7 days.
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from apify_client import ApifyClient

from config import APIFY_API_TOKEN, APIFY_TWITTER_ACTOR, TWEETS_DIR

logger = logging.getLogger(__name__)


class TwitterScraper:
    """Fetches tweets via APIFY for given stock tickers."""

    def __init__(self):
        if not APIFY_API_TOKEN:
            raise ValueError("APIFY_API_TOKEN not set in .env")
        self.client = ApifyClient(APIFY_API_TOKEN)

    def fetch_tweets_for_ticker(
        self,
        ticker: str,
        days: int = 7,
        max_tweets: int = 100,
    ) -> list[dict]:
        """
        Scrape X/Twitter for tweets mentioning a stock ticker in the last N days.
        Returns a list of tweet dicts.
        """
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"${ticker} OR #{ticker} lang:en since:{since_date}"
        logger.info(f"Scraping tweets for {ticker} since {since_date}...")

        run_input = {
            "searchTerms": [query],
            "maxItems": max_tweets,
            "queryType": "Latest",
        }

        tweets = []
        try:
            run = self.client.actor(APIFY_TWITTER_ACTOR).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                logger.warning(f"No dataset returned for {ticker}")
                return tweets

            for item in self.client.dataset(dataset_id).iterate_items():
                tweet = self._normalize_tweet(item, ticker)
                if tweet:
                    tweets.append(tweet)

            logger.info(f"Fetched {len(tweets)} tweets for ${ticker}")

        except Exception as e:
            logger.error(f"APIFY error for {ticker}: {e}")
            tweets = self._load_cached_tweets(ticker)

        self._save_tweets(ticker, tweets)
        return tweets

    def fetch_tweets_for_tickers(
        self,
        tickers: list[str],
        days: int = 7,
        max_tweets_per_ticker: int = 100,
    ) -> dict[str, list[dict]]:
        """Fetch tweets for multiple tickers. Returns dict keyed by ticker."""
        result = {}
        for ticker in tickers:
            result[ticker] = self.fetch_tweets_for_ticker(ticker, days, max_tweets_per_ticker)
            time.sleep(2)  # Brief pause between requests
        return result

    def _normalize_tweet(self, raw: dict, ticker: str) -> Optional[dict]:
        """Normalize raw APIFY tweet data to a consistent format."""
        text = raw.get("full_text") or raw.get("text") or raw.get("tweet_text", "")
        if not text:
            return None

        author = (
            raw.get("author", {}).get("userName")
            or raw.get("user", {}).get("screen_name")
            or raw.get("userName", "")
            or "unknown"
        )
        created_at = (
            raw.get("createdAt")
            or raw.get("created_at")
            or raw.get("tweet_created_at", "")
        )
        likes = (
            raw.get("likeCount")
            or raw.get("favorite_count")
            or raw.get("likes", 0)
        )
        retweets = (
            raw.get("retweetCount")
            or raw.get("retweet_count")
            or raw.get("retweets", 0)
        )
        tweet_id = (
            raw.get("id_str")
            or raw.get("tweet_id")
            or raw.get("id", "")
        )
        url = raw.get("url") or raw.get("tweet_url", "")

        return {
            "id": str(tweet_id),
            "ticker": ticker,
            "author": author,
            "text": text.replace("\n", " ").strip(),
            "created_at": created_at,
            "likes": int(likes) if likes else 0,
            "retweets": int(retweets) if retweets else 0,
            "url": url,
            "engagement": (int(likes or 0) + int(retweets or 0) * 2),
        }

    def _save_tweets(self, ticker: str, tweets: list[dict]):
        path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
        with open(path, "w") as f:
            json.dump(
                {"ticker": ticker, "fetched_at": datetime.utcnow().isoformat(), "tweets": tweets},
                f, indent=2
            )
        logger.info(f"Saved {len(tweets)} tweets for {ticker} to {path}")

    def _load_cached_tweets(self, ticker: str) -> list[dict]:
        path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Loaded cached tweets for {ticker}")
            return data.get("tweets", [])
        return []
