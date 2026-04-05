"""
Social sentiment scraper — multi-source strategy:

PRIMARY:  APIFY Twitter/X — kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest
          Pay-per-result ($0.25/1000 tweets). Works with the free $5 APIFY credit — no rental needed.
FALLBACK: Reddit public JSON API (free, no auth) — used only if APIFY call fails completely.

Twitter/X is the primary source because it gives real-time stock sentiment from actual traders.
Reddit is a solid backup across r/stocks, r/wallstreetbets, r/investing, etc.
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
    Fetches real tweets via APIFY actor:
      kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest

    This actor is pay-per-result ($0.25/1000 tweets) and works with APIFY's
    free $5 credit — no rental/subscription needed.

    Automatically falls back to Reddit if APIFY fails for any reason.
    """

    def __init__(self):
        if not APIFY_API_TOKEN:
            raise ValueError("APIFY_API_TOKEN not set in .env")
        try:
            from apify_client import ApifyClient
            self.client = ApifyClient(APIFY_API_TOKEN)
        except ImportError:
            raise ValueError("apify-client not installed. Run: pip install apify-client")
        self._fallback = RedditScraper()
        logger.info(f"TwitterScraper initialised with actor: {APIFY_TWITTER_ACTOR}")

    def fetch_tweets_for_ticker(
        self, ticker: str, days: int = 7, max_tweets: int = 100
    ) -> list[dict]:
        """Fetch tweets mentioning $TICKER from the last `days` days."""
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        # Twitter advanced search syntax — works with kaitoeasyapi actor
        query = f"${ticker} OR #{ticker} since:{since_date}"
        run_input = {
            "searchTerms": [query],   # kaitoeasyapi actor input field
            "maxItems": max_tweets,   # cap results to control credit usage
            "lang": "en",
        }
        logger.info(f"[APIFY] Searching X for: {query}  (max {max_tweets} tweets)")
        tweets = self._run_actor_with_fallback(run_input, ticker, days, max_tweets)
        self._save_tweets(ticker, tweets)
        return tweets

    def fetch_tweets_for_tickers(
        self, tickers: list[str], days: int = 7, max_tweets_per_ticker: int = 100
    ) -> dict[str, list[dict]]:
        result = {}
        for ticker in tickers:
            result[ticker] = self.fetch_tweets_for_ticker(ticker, days, max_tweets_per_ticker)
            time.sleep(1)  # small gap between actor calls
        return result

    def fetch_creator_sentiment(
        self, tickers: list[str], creators: list[str] = None,
        days: int = 7, max_tweets_per_creator: int = 20,
    ) -> dict:
        """
        Track 100 finance X creators' sentiment on the given tickers.
        Fetches real tweets via APIFY, falls back to Reddit if needed.
        """
        logger.info(f"Fetching creator sentiment for {tickers} via APIFY X scraper...")
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        # Build multi-ticker search queries
        search_terms = []
        for ticker in tickers:
            search_terms.append(f"${ticker} OR #{ticker} since:{since_date}")

        run_input = {
            "searchTerms": search_terms,
            "maxItems": max_tweets_per_creator * len(tickers),
            "lang": "en",
        }

        all_tweets: list[dict] = []
        try:
            run = self.client.actor(APIFY_TWITTER_ACTOR).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            if dataset_id:
                for item in self.client.dataset(dataset_id).iterate_items():
                    for ticker in tickers:
                        if f"${ticker}" in item.get("full_text", "") or f"#{ticker}" in item.get("full_text", ""):
                            t = self._normalize_tweet(item, ticker)
                            if t:
                                all_tweets.append(t)
                                break
                    else:
                        # default to first ticker if none found explicitly
                        t = self._normalize_tweet(item, tickers[0])
                        if t:
                            all_tweets.append(t)
            logger.info(f"[APIFY] creator sentiment: {len(all_tweets)} tweets fetched")
        except Exception as e:
            logger.warning(f"APIFY creator sentiment failed: {e}. Falling back to Reddit.")
            return self._fallback.fetch_creator_sentiment(tickers, creators, days, max_tweets_per_creator * 5)

        # Bucket by ticker and compute creator stats
        all_posts_by_ticker: dict[str, list[dict]] = {t: [] for t in tickers}
        creator_stats: dict[str, dict] = {}

        for tweet in all_tweets:
            ticker = tweet["ticker"]
            all_posts_by_ticker.setdefault(ticker, []).append(tweet)
            author = tweet.get("author", "unknown")
            if author == "unknown":
                continue
            if author not in creator_stats:
                creator_stats[author] = {"total_tweets": 0, "total_engagement": 0,
                                          "tickers_mentioned": [], "platform": "twitter"}
            creator_stats[author]["total_tweets"] += 1
            creator_stats[author]["total_engagement"] += tweet.get("engagement", 0)
            if ticker not in creator_stats[author]["tickers_mentioned"]:
                creator_stats[author]["tickers_mentioned"].append(ticker)

        top_creators = sorted(
            [{"creator": k, **v} for k, v in creator_stats.items()],
            key=lambda x: x["total_engagement"],
            reverse=True,
        )[:20]

        result = {
            "source": "twitter",
            "creators_tracked": len(creator_stats),
            "creator_tweets": all_posts_by_ticker,
            "creator_stats": creator_stats,
            "top_creators": top_creators,
            "fetched_at": datetime.utcnow().isoformat(),
        }

        path = Path(TWEETS_DIR) / "creator_sentiment.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _run_actor_with_fallback(
        self, run_input: dict, ticker: str, days: int, max_tweets: int
    ) -> list[dict]:
        """Run the APIFY actor; fall back to Reddit if it fails."""
        try:
            run = self.client.actor(APIFY_TWITTER_ACTOR).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                raise ValueError("No dataset ID returned by actor")
            tweets = []
            for item in self.client.dataset(dataset_id).iterate_items():
                t = self._normalize_tweet(item, ticker)
                if t:
                    tweets.append(t)
            if tweets:
                logger.info(f"[APIFY] {len(tweets)} tweets fetched for ${ticker}")
                return tweets
            logger.warning(f"[APIFY] 0 tweets returned for ${ticker} — falling back to Reddit")
        except Exception as e:
            logger.warning(f"[APIFY] error for ${ticker}: {e} — falling back to Reddit")
        return self._fallback.fetch_for_ticker(ticker, days, max_tweets)

    def _normalize_tweet(self, raw: dict, ticker: str) -> Optional[dict]:
        """Normalize kaitoeasyapi actor output to a standard tweet dict."""
        # kaitoeasyapi returns: full_text / text, author (obj or str),
        # likeCount / favorite_count, retweetCount / retweet_count,
        # createdAt / created_at, url, id / id_str
        text = (raw.get("full_text") or raw.get("text") or
                raw.get("tweet_text") or raw.get("content", ""))
        if not text:
            return None

        # Author can be a nested object or a plain string
        author_field = raw.get("author") or raw.get("user") or {}
        if isinstance(author_field, dict):
            author = (author_field.get("userName") or author_field.get("screen_name")
                      or author_field.get("name") or "unknown")
        else:
            author = str(author_field) or raw.get("userName", "unknown")

        likes = int(raw.get("likeCount") or raw.get("favorite_count") or 0)
        retweets = int(raw.get("retweetCount") or raw.get("retweet_count") or 0)
        tweet_id = str(raw.get("id_str") or raw.get("id") or raw.get("tweetId", ""))
        created_at = (raw.get("createdAt") or raw.get("created_at") or
                      raw.get("timestamp", ""))
        url = raw.get("url") or raw.get("tweetUrl") or (
            f"https://x.com/{author}/status/{tweet_id}" if tweet_id else ""
        )

        return {
            "id": tweet_id,
            "ticker": ticker,
            "author": author,
            "text": text.replace("\n", " ").strip()[:500],
            "created_at": created_at,
            "likes": likes,
            "retweets": retweets,
            "url": url,
            "engagement": likes + retweets * 2,
            "sentiment": "neutral",
            "source": "twitter",
        }

    def _save_tweets(self, ticker: str, tweets: list[dict]):
        path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
        source = tweets[0]["source"] if tweets else "unknown"
        with open(path, "w") as f:
            json.dump(
                {"ticker": ticker, "source": source,
                 "fetched_at": datetime.utcnow().isoformat(), "tweets": tweets},
                f, indent=2,
            )
