"""
Main data pipeline: orchestrates SEC fetching → tweet scraping → RAG indexing.

Run this once to populate the knowledge base before starting the chatbot.
Can also be scheduled to refresh the data periodically.
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
    ) -> dict:
        """
        Full pipeline:
        1. Fetch top N SEC insider trades (last 24h)
        2. For each ticker, fetch tweets (last 7 days)
        3. Index everything into ChromaDB

        Returns a summary dict with the collected data.
        """
        console.rule("[bold cyan]CrowdWisdomTrading Data Pipeline")
        start_time = datetime.utcnow()

        # ── Step 1: SEC data ─────────────────────────────────────────────
        console.print("\n[bold yellow]Step 1/3:[/] Fetching SEC insider trading data...")
        trades = []

        if refresh_data:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                task = p.add_task("Fetching Form 4 filings from SEC EDGAR...", total=None)
                trades = self.sec_fetcher.get_top_insider_trades(
                    hours=24, top_n=top_n_trades
                )
                p.update(task, description=f"Found {len(trades)} trades")
        else:
            trades = self.sec_fetcher._load_cached_data()
            console.print(f"  Using cached data: {len(trades)} trades")

        if not trades:
            console.print("[red]No insider trades found. Check SEC EDGAR availability.[/]")
            return {"error": "No SEC data available", "trades": [], "tweets": {}}

        self._print_trades_table(trades)
        tickers = list({t["ticker"] for t in trades if t.get("ticker")})
        console.print(f"\n[green]Tickers to monitor:[/] {', '.join(f'${t}' for t in tickers)}")

        # ── Step 2: Tweet scraping ───────────────────────────────────────
        console.print("\n[bold yellow]Step 2/3:[/] Scraping X/Twitter via APIFY...")
        tweet_data = {}

        try:
            scraper = TwitterScraper()
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                for ticker in tickers:
                    task = p.add_task(f"Fetching tweets for ${ticker}...", total=None)
                    tweets = scraper.fetch_tweets_for_ticker(
                        ticker, days=tweet_days, max_tweets=max_tweets_per_ticker
                    )
                    tweet_data[ticker] = tweets
                    p.update(task, description=f"${ticker}: {len(tweets)} tweets fetched")
        except ValueError as e:
            console.print(f"[yellow]APIFY not configured: {e}[/]")
            console.print("[yellow]Loading cached tweet data if available...[/]")
            tweet_data = self._load_cached_tweets(tickers)

        total_tweets = sum(len(v) for v in tweet_data.values())
        console.print(f"[green]Total tweets collected:[/] {total_tweets}")

        # ── Step 3: RAG indexing ─────────────────────────────────────────
        console.print("\n[bold yellow]Step 3/3:[/] Indexing into ChromaDB...")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task("Indexing SEC trades...", total=None)
            sec_chunks = self.indexer.index_sec_trades(trades)
            p.update(task, description=f"Indexed {sec_chunks} SEC trade chunks")

            task2 = p.add_task("Indexing tweets...", total=None)
            tweet_chunks = self.indexer.index_tweets(tweet_data)
            p.update(task2, description=f"Indexed {tweet_chunks} tweet chunks")

        stats = self.indexer.get_stats()
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        console.print(f"\n[bold green]Pipeline complete![/] ({elapsed:.1f}s)")
        console.print(f"  SEC chunks in DB: {stats.get('sec_insider_trades', 0)}")
        console.print(f"  Tweet chunks in DB: {stats.get('tweets', 0)}")

        return {
            "trades": trades,
            "tweets": tweet_data,
            "tickers": tickers,
            "stats": stats,
            "elapsed_seconds": elapsed,
        }

    def _print_trades_table(self, trades: list[dict]):
        table = Table(title="Top SEC Insider Trades (Last 24h)", style="cyan")
        table.add_column("Ticker", style="bold yellow")
        table.add_column("Company")
        table.add_column("Insider")
        table.add_column("Type", style="bold")
        table.add_column("Value", style="bold green")
        table.add_column("Date")

        for t in trades:
            value_str = f"${t.get('total_value', 0):,.0f}"
            type_style = "[green]BUY[/]" if t.get("transaction_type") == "BUY" else "[red]SELL[/]"
            table.add_row(
                t.get("ticker", "N/A"),
                t.get("company", "")[:30],
                t.get("insider_name", "")[:20],
                type_style,
                value_str,
                t.get("transaction_date", ""),
            )
        console.print(table)

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
