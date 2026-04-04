"""
RAG indexer: chunks SEC filings and tweets, embeds them, stores in ChromaDB.

Chunking strategy:
  - SEC Form 4 transactions: 1 transaction = 1 chunk (each trade is atomic context)
    Metadata: ticker, insider, date, value, type
  - Tweets: 1 tweet = 1 chunk (each tweet is self-contained)
    Metadata: ticker, author, date, engagement score
  - Chunks are enriched with a human-readable summary as the embedded text,
    so semantic search hits on meaning rather than raw field names.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

SEC_COLLECTION = "sec_insider_trades"
TWEETS_COLLECTION = "tweets"


class RAGIndexer:
    """Indexes SEC and tweet data into ChromaDB for semantic retrieval."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"RAG indexer initialized. DB path: {CHROMA_DB_PATH}")

    def _get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # SEC filing indexing
    # ------------------------------------------------------------------ #

    def index_sec_trades(self, trades: list[dict]) -> int:
        """
        Index SEC insider trading transactions.
        Each transaction becomes one chunk with a descriptive summary.
        Returns the number of chunks added.
        """
        collection = self._get_or_create_collection(SEC_COLLECTION)

        documents, metadatas, ids = [], [], []
        for i, trade in enumerate(trades):
            doc_text = self._sec_trade_to_text(trade)
            meta = {
                "ticker": trade.get("ticker", ""),
                "company": trade.get("company", ""),
                "insider_name": trade.get("insider_name", ""),
                "insider_title": trade.get("insider_title", ""),
                "transaction_type": trade.get("transaction_type", ""),
                "total_value": float(trade.get("total_value", 0)),
                "shares": float(trade.get("shares", 0)),
                "price_per_share": float(trade.get("price_per_share", 0)),
                "transaction_date": trade.get("transaction_date", ""),
                "source_url": trade.get("source_url", ""),
                "data_type": "sec_trade",
                "indexed_at": datetime.utcnow().isoformat(),
            }
            uid = f"sec_{trade.get('ticker','')}_{trade.get('insider_name','')}_{trade.get('transaction_date','')}"
            uid = uid.replace(" ", "_").replace("/", "-")[:100] + f"_{i}"

            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(uid)

        if documents:
            # Upsert to avoid duplicates on re-index
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} SEC trade chunks")

        return len(documents)

    def _sec_trade_to_text(self, trade: dict) -> str:
        """Convert a SEC trade dict to a human-readable text chunk for embedding."""
        value = trade.get("total_value", 0)
        value_str = f"${value:,.0f}" if value >= 1 else "unknown value"
        return (
            f"SEC Form 4 insider trade: {trade.get('insider_name', 'Unknown insider')} "
            f"({trade.get('insider_title', 'insider')}) at {trade.get('company', 'unknown company')} "
            f"(ticker: {trade.get('ticker', 'N/A')}) made a {trade.get('transaction_type', 'transaction')} "
            f"of {trade.get('shares', 0):,.0f} shares at ${trade.get('price_per_share', 0):.2f}/share, "
            f"totaling {value_str} on {trade.get('transaction_date', 'unknown date')}."
        )

    # ------------------------------------------------------------------ #
    # Tweet indexing
    # ------------------------------------------------------------------ #

    def index_tweets(self, tweet_data: dict[str, list[dict]]) -> int:
        """
        Index tweets per ticker.
        Each tweet is one chunk.
        Returns total chunks added.
        """
        collection = self._get_or_create_collection(TWEETS_COLLECTION)

        total = 0
        for ticker, tweets in tweet_data.items():
            documents, metadatas, ids = [], [], []
            for i, tweet in enumerate(tweets):
                doc_text = self._tweet_to_text(tweet, ticker)
                meta = {
                    "ticker": ticker,
                    "author": tweet.get("author", ""),
                    "created_at": tweet.get("created_at", ""),
                    "likes": int(tweet.get("likes", 0)),
                    "retweets": int(tweet.get("retweets", 0)),
                    "engagement": int(tweet.get("engagement", 0)),
                    "url": tweet.get("url", ""),
                    "data_type": "tweet",
                    "indexed_at": datetime.utcnow().isoformat(),
                }
                uid = f"tweet_{ticker}_{tweet.get('id', i)}"[:100]

                documents.append(doc_text)
                metadatas.append(meta)
                ids.append(uid)

            if documents:
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                logger.info(f"Indexed {len(documents)} tweets for ${ticker}")
                total += len(documents)

        return total

    def _tweet_to_text(self, tweet: dict, ticker: str) -> str:
        """Convert a tweet dict to a text chunk for embedding."""
        return (
            f"Tweet about ${ticker} by @{tweet.get('author', 'unknown')} "
            f"on {tweet.get('created_at', 'unknown date')}: "
            f"{tweet.get('text', '')} "
            f"[{tweet.get('likes', 0)} likes, {tweet.get('retweets', 0)} retweets]"
        )

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Return index statistics."""
        stats = {}
        for name in [SEC_COLLECTION, TWEETS_COLLECTION]:
            try:
                col = self._get_or_create_collection(name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats

    def clear_all(self):
        """Clear all collections (for re-indexing)."""
        for name in [SEC_COLLECTION, TWEETS_COLLECTION]:
            try:
                self.client.delete_collection(name)
                logger.info(f"Cleared collection: {name}")
            except Exception:
                pass
