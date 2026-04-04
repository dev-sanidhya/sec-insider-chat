"""
Main data pipeline: orchestrates SEC fetching → tweet scraping → creator sentiment → RAG indexing.

Pipeline stages:
  1. Fetch top 5 SEC Form 4 insider trades (last 24h) — uses EDGAR RSS + filter logic
     from wescules/insider-trading-analyzer (cluster buys, big money, repeated buyers)
  2. Ticker-based tweet scraping via APIFY (last 7 days)
  3. Creator-based sentiment: track 100 financial X creators for these tickers
     ("sentiment analysis for 100 X user creators + reply plan")
  4. Index all data into ChromaDB for RAG

Run once to populate the knowledge base, then re-run to refresh.
Can be scheduled: python -m flow.pipeline
"""
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import SEC_DATA_DIR
from tools.sec_tools import SECInsiderTradingFetcher
from tools.social_tools import TwitterScraper  # uses APIFY with StockTwits fallback
from rag.indexer import RAGIndexer

logger = logging.getLogger(__name__)

# Force UTF-8 on Windows to avoid cp1252 crash with Rich's Unicode spinner chars
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()


class DataPipeline:
    """Orchestrates the full data collection and indexing pipeline."""

    def __init__(self):
        self.sec_fetcher = SECInsiderTradingFetcher()
        self.indexer = RAGIndexer()

    def run(
        self,
        refresh_data: bool = True,
        top_n_trades: int = 5,
        tweet_days: int = 7,
        max_tweets_per_ticker: int = 100,
        track_creators: bool = True,
    ) -> dict:
        """
        Full pipeline:
        1. Fetch top N SEC insider trades (last 24h) with signal classification
        2. Ticker-based tweet scraping (last 7 days)
        3. 100 X creator sentiment + reply plan (the assignment's "100 X users creators")
        4. Index everything into ChromaDB

        Returns a summary dict with all collected data.
        """
        console.rule("[bold cyan]CrowdWisdomTrading Data Pipeline")
        start_time = datetime.utcnow()

        # ── Step 1: SEC data ─────────────────────────────────────────────
        console.print("\n[bold yellow]Step 1/4:[/] Fetching SEC insider trading data...")
        trades = []

        if refresh_data:
            logger.info("Fetching Form 4 filings + running signal filters...")
            trades = self.sec_fetcher.get_top_insider_trades(hours=24, top_n=top_n_trades)
            logger.info(f"Found {len(trades)} trades with signals")
        else:
            trades = self.sec_fetcher._load_cached_data()
            logger.info(f"Using cached data: {len(trades)} trades")

        if not trades:
            logger.error("No insider trades found. Check SEC EDGAR availability.")
            return {"error": "No SEC data available", "trades": [], "tweets": {}}

        self._print_trades_table(trades)
        tickers = list({t["ticker"] for t in trades if t.get("ticker")})
        logger.info(f"Tickers to monitor: {', '.join(tickers)}")

        # ── Step 2: Ticker tweet scraping ────────────────────────────────
        logger.info("Step 2/4: Scraping X/Twitter (ticker-based) via APIFY...")
        tweet_data = {}
        creator_data = {}

        try:
            scraper = TwitterScraper()

            for ticker in tickers:
                logger.info(f"Fetching tweets for ${ticker}...")
                tweets = scraper.fetch_tweets_for_ticker(
                    ticker, days=tweet_days, max_tweets=max_tweets_per_ticker
                )
                tweet_data[ticker] = tweets
                logger.info(f"  ${ticker}: {len(tweets)} tweets fetched")

            # ── Step 3: 100 creator sentiment ─────────────────────────────
            if track_creators:
                logger.info("Step 3/4: Tracking 100 X financial creators...")
                creator_data = scraper.fetch_creator_sentiment(
                    tickers=tickers,
                    days=tweet_days,
                    max_tweets_per_creator=20,
                )
                n_creators = creator_data.get("creators_tracked", 0)
                top = creator_data.get("top_creators", [])
                logger.info(f"Tracked {n_creators} creators, {len(top)} active on these tickers")
                self._print_creator_table(top[:5])
            else:
                logger.info("Step 3/4: Creator tracking skipped")

        except ValueError as e:
            logger.warning(f"APIFY not configured: {e}")
            tweet_data = self._load_cached_tweets(tickers)

        total_tweets = sum(len(v) for v in tweet_data.values())
        logger.info(f"Total ticker tweets collected: {total_tweets}")

        # ── Step 4: RAG indexing ─────────────────────────────────────────
        logger.info("Step 4/4: Indexing into ChromaDB...")
        sec_chunks = self.indexer.index_sec_trades(trades)
        logger.info(f"Indexed {sec_chunks} SEC trade chunks")

        tweet_chunks = self.indexer.index_tweets(tweet_data)
        logger.info(f"Indexed {tweet_chunks} tweet chunks")

        if creator_data:
            creator_tweets_by_ticker = creator_data.get("creator_tweets", {})
            creator_chunks = self.indexer.index_tweets(creator_tweets_by_ticker)
            logger.info(f"Indexed {creator_chunks} creator tweet chunks")

        stats = self.indexer.get_stats()
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        console.print(f"\n[bold green]Pipeline complete![/] ({elapsed:.1f}s)")
        console.print(f"  SEC chunks in DB:   {stats.get('sec_insider_trades', 0)}")
        console.print(f"  Tweet chunks in DB: {stats.get('tweets', 0)}")

        return {
            "trades": trades,
            "tweets": tweet_data,
            "creator_data": creator_data,
            "tickers": tickers,
            "stats": stats,
            "elapsed_seconds": elapsed,
        }

    def _print_creator_table(self, top_creators: list[dict]):
        if not top_creators:
            return
        table = Table(title="Top Active X Creators on These Tickers", style="magenta")
        table.add_column("Creator", style="bold cyan")
        table.add_column("Tweets")
        table.add_column("Total Engagement", style="bold green")
        table.add_column("Tickers Mentioned")
        for c in top_creators:
            table.add_row(
                f"@{c.get('creator', '')}",
                str(c.get("total_tweets", 0)),
                str(c.get("total_engagement", 0)),
                ", ".join(c.get("tickers_mentioned", [])),
            )
        console.print(table)

    def _print_trades_table(self, trades: list[dict]):
        table = Table(title="Top SEC Insider Trades (Last 24h)", style="cyan")
        table.add_column("Ticker", style="bold yellow")
        table.add_column("Company")
        table.add_column("Insider")
        table.add_column("Type", style="bold")
        table.add_column("Value", style="bold green")
        table.add_column("Signal")
        table.add_column("Date")

        signal_colors = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}
        for t in trades:
            value_str = f"${t.get('total_value', 0):,.0f}"
            type_style = "[green]BUY[/]" if t.get("transaction_type") == "BUY" else "[red]SELL[/]"
            sig = t.get("signal", {})
            sig_label = sig.get("signal", "N/A") if isinstance(sig, dict) else "N/A"
            sig_color = signal_colors.get(sig_label, "white")
            cluster = " [C]" if t.get("cluster_buy") else ""
            table.add_row(
                t.get("ticker", "N/A"),
                t.get("company", "")[:28],
                t.get("insider_name", "")[:18],
                type_style,
                value_str,
                f"[{sig_color}]{sig_label}{cluster}[/]",
                t.get("transaction_date", ""),
            )
        console.print(table)
        console.print("[dim]  [C] = Cluster buy (multiple insiders)[/]")

    def _load_cached_tweets(self, tickers: list[str]) -> dict:
        """Load cached tweet data from disk."""
        from config import TWEETS_DIR
        result = {}
        for ticker in tickers:
            path = Path(TWEETS_DIR) / f"{ticker}_tweets.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                result[ticker] = data.get("tweets", [])
            else:
                result[ticker] = []
        return result
