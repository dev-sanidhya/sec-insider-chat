"""
Social sentiment scraper — multi-source strategy:

PRIMARY:   Reddit public API (free, no auth, r/stocks + r/wallstreetbets + r/investing)
SECONDARY: StockTwits (free API — falls back if rate-limited)
OPTIONAL:  APIFY Twitter (requires paid actor rental on APIFY)

Reddit gives real trader discussions about stock tickers across multiple subreddits.
StockTwits gives stock-specific social posts with explicit Bullish/Bearish labels.
APIFY Twitter is preserved for when the user upgrades their APIFY subscription.
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import APIFY_API_TOKEN, APIFY_TWITTER_ACTOR, TWEETS_DIR

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"

# Subreddits to search for stock discussions (100 creator equivalent tracking)
FINANCE_SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "StockMarket", "options",
    "SecurityAnalysis", "Daytrading", "ValueInvesting", "pennystocks",
    "finance",
]


# ─────────────────────────────────────────────────────────────────────────────
# Reddit scraper (free, no auth, public JSON API)
# ─────────────────────────────────────────────────────────────────────────────

class RedditScraper:
    """
    Scrapes Reddit for stock ticker discussions across finance subreddits.
    Uses Reddit's public JSON API — completely free, no API key required.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CrowdWisdomTrading/1.0 (financial research bot; contact: research@crowdwisdomtrading.com)"
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def _get(self, url: str, params: dict = None) -> dict:
        resp = self.session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def fetch_for_ticker(
        self,
        ticker: str,
        days: int = 7,
        max_posts: int = 100,
        subreddits: list[str] = None,
    ) -> list[dict]:
        """
        Fetch Reddit posts mentioning a stock ticker from multiple subreddits.
        Returns normalized tweet-compatible dicts.
        """
        if subreddits is None:
            subreddits = FINANCE_SUBREDDITS

        logger.info(f"Fetching Reddit posts for ${ticker} from {len(subreddits)} subreddits...")
        all_posts = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        for subreddit in subreddits:
            try:
                data = self._get(
                    f"{REDDIT_BASE}/r/{subreddit}/search.json",
                    params={"q": ticker, "restrict_sr": "1", "sort": "new", "limit": 25, "t": "week"},
                )
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    normalized = self._normalize_post(post.get("data", {}), ticker, subreddit)
                    if normalized:
                        # Filter by date
                        try:
                            dt = datetime.fromtimestamp(
                                post["data"].get("created_utc", 0), tz=timezone.utc
                            )
                            if dt >= cutoff:
                                all_posts.append(normalized)
                        except Exception:
                            all_posts.append(normalized)

                time.sleep(0.5)  # Reddit rate limit: ~1 req/sec

            except Exception as e:
                logger.debug(f"Reddit error for r/{subreddit} + {ticker}: {e}")

            if len(all_posts) >= max_posts:
                break

        all_posts = all_posts[:max_posts]
        logger.info(f"Fetched {len(all_posts)} Reddit posts for ${ticker}")
        self._save(ticker, all_posts)
        return all_posts

    def fetch_for_tickers(
        self, tickers: list[str], days: int = 7, max_per_ticker: int = 100
    ) -> dict[str, list[dict]]:
        result = {}
        for ticker in tickers:
            result[ticker] = self.fetch_for_ticker(ticker, days, max_per_ticker)
            time.sleep(1)
        return result

    def fetch_creator_sentiment(
        self,
        tickers: list[str],
        creators: list[str] = None,
        days: int = 7,
        max_per_ticker: int = 200,
    ) -> dict:
        """
        Fetch Reddit posts and identify the most active/influential users
        posting about the given tickers across finance subreddits.
        This is the '100 X creators' equivalent on Reddit.
        """
        logger.info(f"Fetching Reddit creator sentiment for {tickers}...")

        all_posts_by_ticker: dict[str, list[dict]] = {}
        creator_stats: dict[str, dict] = {}

        for ticker in tickers:
            posts = self.fetch_for_ticker(ticker, days, max_per_ticker)
            all_posts_by_ticker[ticker] = posts

            for post in posts:
                author = post.get("author", "unknown")
                if author in ("[deleted]", "AutoModerator", "unknown"):
                    continue
                if author not in creator_stats:
                    creator_stats[author] = {
                        "total_tweets": 0,
                        "total_engagement": 0,
                        "tickers_mentioned": [],
                        "subreddits": [],
                    }
                creator_stats[author]["total_tweets"] += 1
                creator_stats[author]["total_engagement"] += post.get("engagement", 0)
                if ticker not in creator_stats[author]["tickers_mentioned"]:
                    creator_stats[author]["tickers_mentioned"].append(ticker)
                sr = post.get("subreddit", "")
                if sr and sr not in creator_stats[author]["subreddits"]:
                    creator_stats[author]["subreddits"].append(sr)

        top_creators = sorted(
            [{"creator": k, **v} for k, v in creator_stats.items()],
            key=lambda x: x["total_engagement"],
            reverse=True,
        )[:20]

        result = {
            "source": "reddit",
            "creators_tracked": len(creator_stats),
            "subreddits_searched": FINANCE_SUBREDDITS,
            "creator_tweets": all_posts_by_ticker,
            "creator_stats": creator_stats,
            "top_creators": top_creators,
            "fetched_at": datetime.utcnow().isoformat(),
        }

        path = Path(TWEETS_DIR) / "creator_sentiment.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"Reddit creator sentiment: {len(creator_stats)} unique creators "
            f"across {len(tickers)} tickers, {len(FINANCE_SUBREDDITS)} subreddits"
        )
        return result

    def _normalize_post(self, raw: dict, ticker: str, subreddit: str) -> Optional[dict]:
        title = raw.get("title", "")
        selftext = raw.get("selftext", "")
        text = f"{title}. {selftext}".strip(" .")
        if not text or text == ".":
            return None

        score = raw.get("score", 0)
        num_comments = raw.get("num_comments", 0)
        created_utc = raw.get("created_utc", 0)

        try:
            created_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
        except Exception:
            created_at = ""

        return {
            "id": str(raw.get("id", "")),
            "ticker": ticker,
            "author": raw.get("author", "unknown"),
            "text": text[:500],
            "created_at": created_at,
            "likes": score,
            "retweets": num_comments,  # comments = engagement equivalent
            "url": f"https://reddit.com{raw.get('permalink', '')}",
            "engagement": score + num_comments * 2,
            "sentiment": "neutral",
            "source": "reddit",
            "subreddit": subreddit,
        }

    def _save(self, ticker: str, posts: list[dict]):
        path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
        with open(path, "w") as f:
            json.dump(
                {"ticker": ticker, "source": "reddit",
                 "fetched_at": datetime.utcnow().isoformat(), "tweets": posts},
                f, indent=2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TwitterScraper — APIFY with Reddit fallback
# ─────────────────────────────────────────────────────────────────────────────

class TwitterScraper:
    """
    Tries APIFY Twitter actor first.
    Automatically falls back to Reddit (free) if APIFY actor requires paid rental.
    """

    def __init__(self):
        if not APIFY_API_TOKEN:
            raise ValueError("APIFY_API_TOKEN not set in .env")
        try:
            from apify_client import ApifyClient
            self.client = ApifyClient(APIFY_API_TOKEN)
        except ImportError:
            raise ValueError("apify-client not installed.")
        self._fallback = RedditScraper()

    def fetch_tweets_for_ticker(
        self, ticker: str, days: int = 7, max_tweets: int = 100
    ) -> list[dict]:
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"${ticker} OR #{ticker} lang:en since:{since_date}"
        run_input = {"queries": [query], "maxTweets": max_tweets, "language": "en"}
        tweets = self._run_actor_with_fallback(run_input, ticker, days, max_tweets)
        self._save_tweets(ticker, tweets)
        return tweets

    def fetch_tweets_for_tickers(
        self, tickers: list[str], days: int = 7, max_tweets_per_ticker: int = 100
    ) -> dict[str, list[dict]]:
        result = {}
        for ticker in tickers:
            result[ticker] = self.fetch_tweets_for_ticker(ticker, days, max_tweets_per_ticker)
            time.sleep(1)
        return result

    def fetch_creator_sentiment(
        self, tickers: list[str], creators: list[str] = None,
        days: int = 7, max_tweets_per_creator: int = 20,
    ) -> dict:
        logger.info("Using Reddit for creator sentiment (APIFY actor requires paid rental)")
        return self._fallback.fetch_creator_sentiment(
            tickers, creators, days, max_tweets_per_creator * 5
        )

    def _run_actor_with_fallback(
        self, run_input: dict, ticker: str, days: int, max_tweets: int
    ) -> list[dict]:
        try:
            run = self.client.actor(APIFY_TWITTER_ACTOR).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                raise ValueError("No dataset returned")
            tweets = []
            for item in self.client.dataset(dataset_id).iterate_items():
                t = self._normalize_tweet(item, ticker)
                if t:
                    tweets.append(t)
            if tweets:
                logger.info(f"APIFY returned {len(tweets)} tweets for ${ticker}")
                return tweets
            raise ValueError("0 tweets returned")
        except Exception as e:
            if any(word in str(e).lower() for word in ["paid", "rent", "subscribe", "trial"]):
                logger.warning(f"APIFY actor not available (paid plan required). Using Reddit for ${ticker}.")
            else:
                logger.warning(f"APIFY failed: {e}. Using Reddit for ${ticker}.")
            return self._fallback.fetch_for_ticker(ticker, days, max_tweets)

    def _normalize_tweet(self, raw: dict, ticker: str) -> Optional[dict]:
        text = raw.get("full_text") or raw.get("text") or raw.get("tweet_text", "")
        if not text:
            return None
        author = (
            raw.get("author", {}).get("userName")
            or raw.get("user", {}).get("screen_name")
            or raw.get("userName", "unknown")
        )
        likes = raw.get("likeCount") or raw.get("favorite_count") or 0
        retweets = raw.get("retweetCount") or raw.get("retweet_count") or 0
        return {
            "id": str(raw.get("id_str") or raw.get("id", "")),
            "ticker": ticker,
            "author": author,
            "text": text.replace("\n", " ").strip(),
            "created_at": raw.get("createdAt") or raw.get("created_at", ""),
            "likes": int(likes),
            "retweets": int(retweets),
            "url": raw.get("url", ""),
            "engagement": int(likes) + int(retweets) * 2,
            "sentiment": "neutral",
            "source": "twitter",
        }

    def _save_tweets(self, ticker: str, tweets: list[dict]):
        path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
        with open(path, "w") as f:
            json.dump(
                {"ticker": ticker, "fetched_at": datetime.utcnow().isoformat(), "tweets": tweets},
                f, indent=2,
            )
