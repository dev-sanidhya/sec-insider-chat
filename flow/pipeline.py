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
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config import SEC_DATA_DIR
from tools.sec_tools import SECInsiderTradingFetcher
from tools.apify_tools import TwitterScraper
from rag.indexer import RAGIndexer

logger = logging.getLogger(__name__)
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
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                task = p.add_task("Fetching Form 4 filings + running signal filters...", total=None)
                trades = self.sec_fetcher.get_top_insider_trades(hours=24, top_n=top_n_trades)
                p.update(task, description=f"Found {len(trades)} trades with signals")
        else:
            trades = self.sec_fetcher._load_cached_data()
            console.print(f"  Using cached data: {len(trades)} trades")

        if not trades:
            console.print("[red]No insider trades found. Check SEC EDGAR availability.[/]")
            return {"error": "No SEC data available", "trades": [], "tweets": {}}

        self._print_trades_table(trades)
        tickers = list({t["ticker"] for t in trades if t.get("ticker")})
        console.print(f"\n[green]Tickers to monitor:[/] {', '.join(f'${t}' for t in tickers)}")

        # ── Step 2: Ticker tweet scraping ────────────────────────────────
        console.print("\n[bold yellow]Step 2/4:[/] Scraping X/Twitter (ticker-based) via APIFY...")
        tweet_data = {}
        creator_data = {}

        try:
            scraper = TwitterScraper()

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                for ticker in tickers:
                    task = p.add_task(f"Fetching tweets for ${ticker}...", total=None)
                    tweets = scraper.fetch_tweets_for_ticker(
                        ticker, days=tweet_days, max_tweets=max_tweets_per_ticker
                    )
                    tweet_data[ticker] = tweets
                    p.update(task, description=f"${ticker}: {len(tweets)} tweets")

            # ── Step 3: 100 creator sentiment ─────────────────────────────
            if track_creators:
                console.print("\n[bold yellow]Step 3/4:[/] Tracking 100 X financial creators...")
                with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                    task = p.add_task("Fetching creator sentiment for all tickers...", total=None)
                    creator_data = scraper.fetch_creator_sentiment(
                        tickers=tickers,
                        days=tweet_days,
                        max_tweets_per_creator=20,
                    )
                    n_creators = creator_data.get("creators_tracked", 0)
                    top = creator_data.get("top_creators", [])
                    p.update(task, description=f"Tracked {n_creators} creators, {len(top)} active on these tickers")

                self._print_creator_table(creator_data.get("top_creators", [])[:5])
            else:
                console.print("\n[dim]Step 3/4: Creator tracking skipped (track_creators=False)[/]")

        except ValueError as e:
            console.print(f"[yellow]APIFY not configured: {e}[/]")
            console.print("[yellow]Loading cached data if available...[/]")
            tweet_data = self._load_cached_tweets(tickers)

        total_tweets = sum(len(v) for v in tweet_data.values())
        console.print(f"[green]Total ticker tweets collected:[/] {total_tweets}")

        # ── Step 4: RAG indexing ─────────────────────────────────────────
        console.print("\n[bold yellow]Step 4/4:[/] Indexing into ChromaDB...")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task("Indexing SEC trades...", total=None)
            sec_chunks = self.indexer.index_sec_trades(trades)
            p.update(task, description=f"Indexed {sec_chunks} SEC trade chunks")

            task2 = p.add_task("Indexing ticker tweets...", total=None)
            tweet_chunks = self.indexer.index_tweets(tweet_data)
            p.update(task2, description=f"Indexed {tweet_chunks} tweet chunks")

            # Also index creator tweets per ticker
            if creator_data:
                task3 = p.add_task("Indexing creator tweets...", total=None)
                creator_tweets_by_ticker = creator_data.get("creator_tweets", {})
                creator_chunks = self.indexer.index_tweets(creator_tweets_by_ticker)
                p.update(task3, description=f"Indexed {creator_chunks} creator tweet chunks")

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
