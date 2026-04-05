"""
Gradio chatbot UI for the CrowdWisdomTrading RAG system.
"""
import sys
import io
from pathlib import Path

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from rag.indexer import RAGIndexer

# Initialize RAG once at startup
indexer = RAGIndexer()
sec_col = indexer.client.get_or_create_collection("sec_insider_trades")
tweet_col = indexer.client.get_or_create_collection("tweets")


def chat(message: str, history: list) -> str:
    """
    RAG chatbot function — Gradio 6 requires (message, history) signature.
    Retrieves from ChromaDB and returns answers grounded in the data.
    """
    if not message.strip():
        return "Please ask a question about insider trades or social sentiment."

    # Semantic retrieval from both collections
    sec_results = sec_col.query(query_texts=[message], n_results=3)
    tweet_results = tweet_col.query(query_texts=[message], n_results=3)

    sec_docs = sec_results["documents"][0] if sec_results["documents"] else []
    tweet_docs = tweet_results["documents"][0] if tweet_results["documents"] else []
    sec_meta = sec_results["metadatas"][0] if sec_results["metadatas"] else []
    tweet_meta = tweet_results["metadatas"][0] if tweet_results["metadatas"] else []

    parts = []

    if sec_docs:
        parts.append("SEC INSIDER TRADES (from EDGAR Form 4):")
        for i, (doc, meta) in enumerate(zip(sec_docs, sec_meta), 1):
            ticker = meta.get("ticker", "N/A")
            value = meta.get("total_value", 0)
            insider = meta.get("insider_name", "Unknown")
            tx_type = meta.get("transaction_type", "")
            date = meta.get("transaction_date", "")
            parts.append(
                "\n{}. {} | {} | {} | ${:,.0f} | {}".format(
                    i, ticker, insider, tx_type, value, date
                )
            )
            parts.append("   " + doc[:180])

    if tweet_docs:
        parts.append("\nSOCIAL SENTIMENT (X / Twitter via APIFY):")
        for i, (doc, meta) in enumerate(zip(tweet_docs, tweet_meta), 1):
            author = meta.get("author", "unknown")
            likes = meta.get("likes", 0)
            retweets = meta.get("retweets", 0)
            ticker = meta.get("ticker", "")
            parts.append(
                "\n{}. @{} on ${} | {} likes | {} retweets".format(
                    i, author, ticker, likes, retweets
                )
            )
            parts.append("   " + doc[:180])

    if not parts:
        return "No matching data found in the database. Try asking about CRWV, NTRA, or SYRE."

    return "\n".join(parts)


if __name__ == "__main__":
    stats = indexer.get_stats()

    demo = gr.ChatInterface(
        fn=chat,
        title="CrowdWisdomTrading — Insider Intelligence Chatbot",
        description=(
            "Ask questions about **live SEC Form 4 insider trades** and **real X/Twitter sentiment**.\n\n"
            "Data indexed: **{}** SEC trades | **{}** social posts from X\n\n"
            "Answers are grounded ONLY in real fetched data — no hallucinations."
        ).format(
            stats.get("sec_insider_trades", 0),
            stats.get("tweets", 0),
        ),
        examples=[
            "What insider trades happened in CRWV?",
            "What do people on X say about NTRA?",
            "Which insider sold the most shares recently?",
            "Show me social sentiment about SYRE",
            "What is the most recent SEC filing?",
        ],
        run_examples_on_click=False,   # prevents auto-run at load time (fixes the error)
        autofocus=True,
    )

    print("\n" + "=" * 60)
    print("CrowdWisdomTrading RAG Chatbot")
    print("=" * 60)
    print("SEC chunks:   {}".format(stats.get("sec_insider_trades", 0)))
    print("Tweet chunks: {}".format(stats.get("tweets", 0)))
    print("URL:          http://localhost:7860")
    print("=" * 60 + "\n")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
