#!/usr/bin/env python
"""
Demo script showing RAG chatbot in action with real SEC + Twitter data.
Run this to show Gilad how the chatbot answers questions grounded in the data.
"""
import sys
import io
from pathlib import Path

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from rag.indexer import RAGIndexer


def demo_chat():
    """Interactive demo of RAG chatbot."""
    print("\n" + "=" * 80)
    print("CROWDWISDOMTRADING RAG CHATBOT DEMO")
    print("=" * 80)
    print("\nThis chatbot answers ONLY from real SEC insider trading data")
    print("+ social sentiment from X/Twitter (227 tweets indexed)")
    print("=" * 80 + "\n")

    indexer = RAGIndexer()

    # Pre-loaded demo questions
    demo_questions = [
        "What insider trades happened in CRWV?",
        "Who are the top creators discussing these stocks?",
        "What is the social sentiment about insider selling?",
    ]

    for i, question in enumerate(demo_questions, 1):
        print("\n[Demo Question {}]".format(i))
        print("User: {}".format(question))
        print("-" * 80)

        # Retrieve from RAG
        sec_col = indexer.client.get_or_create_collection("sec_insider_trades")
        tweet_col = indexer.client.get_or_create_collection("tweets")

        sec_results = sec_col.query(query_texts=[question], n_results=2)
        tweet_results = tweet_col.query(query_texts=[question], n_results=2)

        sec_docs = sec_results["documents"][0] if sec_results["documents"] else []
        tweet_docs = tweet_results["documents"][0] if tweet_results["documents"] else []
        sec_meta = sec_results["metadatas"][0] if sec_results["metadatas"] else []
        tweet_meta = tweet_results["metadatas"][0] if tweet_results["metadatas"] else []

        # Format response
        response = "\nChatbot Response (based on indexed data):\n"

        if sec_docs:
            response += "\nSEC INSIDER TRADES:\n"
            for j, (doc, meta) in enumerate(zip(sec_docs, sec_meta), 1):
                ticker = meta.get("ticker", "N/A")
                insider = meta.get("insider_name", "Unknown")
                value = meta.get("total_value", 0)
                tx_type = meta.get("transaction_type", "")
                value_str = "${:,.0f}".format(value) if value else "unknown"

                response += "\n{}. {} ({}) in {}\n".format(j, insider, tx_type, ticker)
                response += "   Value: {}\n".format(value_str)
                response += "   Details: {}\n".format(doc[:150])

        if tweet_docs:
            response += "\nSOCIAL SENTIMENT (X/TWITTER):\n"
            for j, (doc, meta) in enumerate(zip(tweet_docs, tweet_meta), 1):
                author = meta.get("author", "unknown")
                engagement = meta.get("engagement", 0)

                response += "\n{}. @{} (engagement: {})\n".format(j, author, engagement)
                response += "   {}\n".format(doc[:150])

        if not sec_docs and not tweet_docs:
            response += "\nNo matching data found in database."

        print(response)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nTo start the interactive Gradio UI, run:")
    print("  python app/chatbot_ui.py")
    print("\nThen open http://localhost:7860 in your browser")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demo_chat()
