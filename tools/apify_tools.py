"""
APIFY Twitter/X scraper tool.

Two scraping modes:
  1. Ticker-based : fetch tweets mentioning stock tickers (last 7 days)
  2. Creator-based: track 100 influential financial X creators and
                    analyze their posts about the relevant tickers
                    ("sentiment analysis for 100 X user creators")
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

# ─────────────────────────────────────────────────────────────────────────────
# 100 influential financial X / Twitter creators to track
# Covers: analysts, hedge fund managers, retail traders, fintwit personalities
# ─────────────────────────────────────────────────────────────────────────────
TOP_100_FINANCE_CREATORS = [
    # Institutional / macro analysts
    "elonmusk", "chamath", "naval", "garrytan", "balajis",
    "BrendanEich", "saylor", "APompliano", "CathieDWood",
    "RaoulGMI", "LukeGromen", "LynAldenContact", "DanielJLacalle",
    "SaraEisen", "carlquintanilla", "andrew_r_sorkin", "beckyquick",
    # Fintwit traders
    "TruthGundersen", "OptionsHawk", "Canaccord", "ukarlewitz",
    "macroalf", "quantian1", "LiveSquawk", "FirstSquawk",
    "zerohedge", "business", "Reuters", "markets",
    "WSJmarkets", "FT", "TheTerminal", "BloombergTV",
    # Retail / individual investors
    "TheStalwart", "ReformedBroker", "DougKass", "Josh_Young_1",
    "BrentBeshore", "PackyM", "EricBalchunas", "atif_mian",
    "GRDecter", "unusual_whales", "cboeoptionshub", "elerianm",
    "NickTimiraos", "JohnKosar", "garyblack00", "StockMarketMoe",
    "TruthGundersen", "WallStreetMemes", "DeItaone", "financialjuice",
    "hmeisler", "andrewthrasher", "mark_ungewitter", "JCParets",
    "RyanDetrick", "SunriseTrader", "allstarcharts", "bespokeinvest",
    "sentimentrader", "Trinhnomics", "darioperkins", "michaellebowitz",
    "MarkYusko", "MorganStanley", "GoldmanSachs", "jpmorgan",
    "BlackRock", "VanguardGroup", "Fidelity", "Schwab",
    "tastytrade", "OptionsAction", "traders_mag", "IBDinvestors",
    "TDAmeritrade", "etradeofficial", "RobinhoodApp", "webullUSA",
    # Crypto / growth investors that cross over
    "VitalikButerin", "cz_binance", "brian_armstrong", "coinbase",
    "novogratz", "BarrySilbert", "RealVision", "PeterSchiff",
    # News aggregators
    "MarketWatch", "YahooFinance", "CNBCFastMoney", "CNBCClosingBell",
    "FoxBusiness", "MorningBrew", "TheMotleyFool", "Barrons",
    "Forbes", "Fortune", "economist", "WSJ",
    # Additional fintwit
    "fundstrat", "tomlee_fundstrat", "KeithMcCullough", "HedgeyeRisk",
    "HedgeyeFT", "KingKenny1973", "ZacharyKarabell", "felixsalmon",
    "ninaandalexis", "SoberLook", "Convertbond", "EddyElfenbein",
]

# Ensure exactly 100
TOP_100_FINANCE_CREATORS = list(dict.fromkeys(TOP_100_FINANCE_CREATORS))[:100]


class TwitterScraper:
    """
    Fetches tweets via APIFY.
    Supports both ticker-based searches and creator-based tracking.
    """

    def __init__(self):
        if not APIFY_API_TOKEN:
            raise ValueError("APIFY_API_TOKEN not set in .env")
        self.client = ApifyClient(APIFY_API_TOKEN)

    # ── Ticker-based search ────────────────────────────────────────────────

    def fetch_tweets_for_ticker(
        self,
        ticker: str,
        days: int = 7,
        max_tweets: int = 100,
    ) -> list[dict]:
        """
        Scrape X for tweets mentioning a stock ticker in the last N days.
        Returns normalized tweet dicts.
        """
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"${ticker} OR #{ticker} lang:en since:{since_date}"
        logger.info(f"Scraping ticker tweets for ${ticker} since {since_date}...")

        # web.harvester/easy-twitter-search-scraper input format
        run_input = {
            "queries": [query],
            "maxTweets": max_tweets,
            "language": "en",
        }

        tweets = self._run_actor(run_input, ticker)
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
            time.sleep(2)
        return result

    # ── Creator-based tracking ─────────────────────────────────────────────

    def fetch_creator_sentiment(
        self,
        tickers: list[str],
        creators: list[str] = None,
        days: int = 7,
        max_tweets_per_creator: int = 20,
    ) -> dict:
        """
        Track 100 influential financial X creators and extract their posts
        about the given tickers in the last N days.

        Returns:
          {
            "creator_tweets": { ticker: [tweets from creators only] },
            "creator_stats":  { creator: { ticker: tweet_count, sentiment } },
            "top_creators":   [ most active creators about these tickers ]
          }
        """
        if creators is None:
            creators = TOP_100_FINANCE_CREATORS

        logger.info(f"Fetching creator sentiment from {len(creators)} X creators for {tickers}...")

        ticker_filter = " OR ".join(f"${t} OR #{t}" for t in tickers)
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        # Build a single search query combining creators + tickers
        # APIFY tweet scraper supports "from:user" queries
        creator_chunks = [creators[i:i+10] for i in range(0, len(creators), 10)]

        all_creator_tweets = []
        for chunk in creator_chunks:
            from_filter = " OR ".join(f"from:{c}" for c in chunk)
            query = f"({from_filter}) ({ticker_filter}) since:{since_date} lang:en"
            run_input = {
                "queries": [query],
                "maxTweets": max_tweets_per_creator * len(chunk),
                "language": "en",
            }
            tweets = self._run_actor(run_input, ticker=tickers[0] if tickers else "")
            all_creator_tweets.extend(tweets)
            time.sleep(1)

        # Organize by ticker
        creator_tweets_by_ticker: dict[str, list] = {t: [] for t in tickers}
        for tweet in all_creator_tweets:
            text_lower = tweet.get("text", "").lower()
            for ticker in tickers:
                if f"${ticker.lower()}" in text_lower or f"#{ticker.lower()}" in text_lower or ticker.lower() in text_lower:
                    t = dict(tweet)
                    t["ticker"] = ticker
                    creator_tweets_by_ticker[ticker].append(t)

        # Creator stats
        creator_stats: dict[str, dict] = {}
        for tweet in all_creator_tweets:
            author = tweet.get("author", "unknown")
            if author not in creator_stats:
                creator_stats[author] = {"total_tweets": 0, "total_engagement": 0, "tickers_mentioned": []}
            creator_stats[author]["total_tweets"] += 1
            creator_stats[author]["total_engagement"] += tweet.get("engagement", 0)
            for ticker in tickers:
                if f"${ticker.lower()}" in tweet.get("text", "").lower():
                    if ticker not in creator_stats[author]["tickers_mentioned"]:
                        creator_stats[author]["tickers_mentioned"].append(ticker)

        # Top creators by engagement
        top_creators = sorted(
            [{"creator": k, **v} for k, v in creator_stats.items()],
            key=lambda x: x["total_engagement"],
            reverse=True,
        )[:20]

        result = {
            "creators_tracked": len(creators),
            "creator_tweets": creator_tweets_by_ticker,
            "creator_stats": creator_stats,
            "top_creators": top_creators,
            "fetched_at": datetime.utcnow().isoformat(),
        }

        # Save to disk
        path = Path(TWEETS_DIR) / "creator_sentiment.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Creator sentiment saved: {len(all_creator_tweets)} total tweets from {len(creator_stats)} creators")

        return result

    def get_creator_reply_plan(
        self,
        creator_tweets: list[dict],
        ticker: str,
    ) -> list[dict]:
        """
        Plan reply suggestions for top-engagement creator tweets about a ticker.
        Returns list of { tweet, suggested_reply_angle, priority }.
        """
        positive_kw = {"bullish", "buy", "long", "moon", "surge", "beat", "strong", "rally",
                       "growth", "profit", "gain", "outperform", "undervalued", "breakout"}
        negative_kw = {"bearish", "sell", "short", "crash", "miss", "weak", "drop", "loss",
                       "fail", "decline", "overvalued", "bubble", "dump"}

        reply_plan = []
        for tweet in sorted(creator_tweets, key=lambda x: x.get("engagement", 0), reverse=True)[:10]:
            text_lower = tweet.get("text", "").lower()
            words = set(text_lower.split())

            if words & positive_kw:
                sentiment = "POSITIVE"
                angle = f"Reinforce with SEC data: insider buys in ${ticker} support this thesis"
            elif words & negative_kw:
                sentiment = "NEGATIVE"
                angle = f"Counter with SEC data: insider activity context for ${ticker}"
            else:
                sentiment = "NEUTRAL"
                angle = f"Add value: share SEC Form 4 insider trading data for ${ticker}"

            reply_plan.append({
                "tweet_author": tweet.get("author", ""),
                "tweet_text": tweet.get("text", "")[:200],
                "engagement": tweet.get("engagement", 0),
                "sentiment": sentiment,
                "suggested_reply_angle": angle,
                "priority": "HIGH" if tweet.get("engagement", 0) > 100 else "MEDIUM",
            })

        return reply_plan

    # ── Internal helpers ───────────────────────────────────────────────────

    def _run_actor(self, run_input: dict, ticker: str = "") -> list[dict]:
        """Run the APIFY Twitter actor and return normalized tweets."""
        tweets = []
        try:
            run = self.client.actor(APIFY_TWITTER_ACTOR).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                logger.warning("No dataset returned from APIFY actor")
                return self._load_cached_tweets(ticker)

            for item in self.client.dataset(dataset_id).iterate_items():
                tweet = self._normalize_tweet(item, ticker)
                if tweet:
                    tweets.append(tweet)

            logger.info(f"APIFY returned {len(tweets)} tweets")
        except Exception as e:
            logger.error(f"APIFY actor error: {e}")
            tweets = self._load_cached_tweets(ticker)
        return tweets

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
            raw.get("createdAt") or raw.get("created_at") or raw.get("tweet_created_at", "")
        )
        likes = raw.get("likeCount") or raw.get("favorite_count") or raw.get("likes", 0)
        retweets = raw.get("retweetCount") or raw.get("retweet_count") or raw.get("retweets", 0)
        tweet_id = raw.get("id_str") or raw.get("tweet_id") or raw.get("id", "")
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
                f, indent=2,
            )
        logger.info(f"Saved {len(tweets)} tweets for {ticker}")

    def _load_cached_tweets(self, ticker: str) -> list[dict]:
        path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
        if path.exists():
            with open(path) as f:
                return json.load(f).get("tweets", [])
        return []
