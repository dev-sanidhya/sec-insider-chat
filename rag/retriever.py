"""
RAG retriever: semantic search over indexed SEC and tweet data.
Returns ranked, formatted context for the LLM.
"""
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K_RESULTS
from rag.indexer import SEC_COLLECTION, TWEETS_COLLECTION

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant context chunks for a user query."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filter_ticker: Optional[str] = None,
        data_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User's natural language question
            top_k: Number of results to return
            filter_ticker: Optional ticker symbol to restrict search
            data_types: Optional list of data types: ["sec_trade", "tweet"]

        Returns:
            List of result dicts with 'text', 'metadata', 'distance', 'collection'
        """
        results = []

        collections_to_search = []
        if not data_types or "sec_trade" in data_types:
            collections_to_search.append(SEC_COLLECTION)
        if not data_types or "tweet" in data_types:
            collections_to_search.append(TWEETS_COLLECTION)

        where_filter = {}
        if filter_ticker:
            where_filter["ticker"] = filter_ticker.upper()

        for col_name in collections_to_search:
            try:
                col = self.client.get_collection(col_name)
                if col.count() == 0:
                    continue

                query_results = col.query(
                    query_texts=[query],
                    n_results=min(top_k, col.count()),
                    where=where_filter if where_filter else None,
                    include=["documents", "metadatas", "distances"],
                )

                docs = query_results.get("documents", [[]])[0]
                metas = query_results.get("metadatas", [[]])[0]
                dists = query_results.get("distances", [[]])[0]

                for doc, meta, dist in zip(docs, metas, dists):
                    results.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "collection": col_name,
                        "relevance_score": max(0.0, 1.0 - dist),  # Convert cosine distance to similarity
                    })

            except Exception as e:
                logger.warning(f"Error querying collection {col_name}: {e}")

        # Sort by relevance (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a prompt-ready context string.
        Groups by data type for clarity.
        """
        if not results:
            return "No relevant data found in the knowledge base."

        sec_chunks = [r for r in results if r["collection"] == SEC_COLLECTION]
        tweet_chunks = [r for r in results if r["collection"] == TWEETS_COLLECTION]

        parts = []

        if sec_chunks:
            parts.append("=== SEC INSIDER TRADING DATA ===")
            for i, r in enumerate(sec_chunks, 1):
                meta = r["metadata"]
                parts.append(
                    f"[{i}] {r['text']}\n"
                    f"    Source: {meta.get('source_url', 'SEC EDGAR')} | "
                    f"Score: {r['relevance_score']:.2f}"
                )

        if tweet_chunks:
            parts.append("\n=== SOCIAL MEDIA (X/TWITTER) DATA ===")
            for i, r in enumerate(tweet_chunks, 1):
                meta = r["metadata"]
                parts.append(
                    f"[{i}] {r['text']}\n"
                    f"    Engagement: {meta.get('engagement', 0)} | "
                    f"Score: {r['relevance_score']:.2f}"
                )

        return "\n".join(parts)

    def get_all_tickers(self) -> list[str]:
        """Return all unique tickers in the knowledge base."""
        tickers = set()
        for col_name in [SEC_COLLECTION, TWEETS_COLLECTION]:
            try:
                col = self.client.get_collection(col_name)
                results = col.get(include=["metadatas"])
                for meta in results.get("metadatas", []):
                    t = meta.get("ticker", "")
                    if t:
                        tickers.add(t)
            except Exception:
                pass
        return sorted(tickers)

    def get_ticker_summary(self, ticker: str) -> dict:
        """Get a quick statistical summary for a ticker from indexed data."""
        summary = {"ticker": ticker, "sec_trades": [], "tweet_count": 0, "total_value": 0}

        try:
            sec_col = self.client.get_collection(SEC_COLLECTION)
            sec_results = sec_col.get(
                where={"ticker": ticker.upper()},
                include=["metadatas"],
            )
            for meta in sec_results.get("metadatas", []):
                summary["sec_trades"].append(meta)
                summary["total_value"] += meta.get("total_value", 0)
        except Exception:
            pass

        try:
            tweet_col = self.client.get_collection(TWEETS_COLLECTION)
            tweet_results = tweet_col.get(
                where={"ticker": ticker.upper()},
                include=["metadatas"],
            )
            summary["tweet_count"] = len(tweet_results.get("metadatas", []))
        except Exception:
            pass

        return summary
