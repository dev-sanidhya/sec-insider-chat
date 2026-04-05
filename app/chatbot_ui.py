"""
Gradio chatbot UI for the CrowdWisdomTrading RAG system.
Allows users to chat with the insider trading + social sentiment data.
"""
import logging
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

logger = logging.getLogger(__name__)

# Initialize RAG
indexer = RAGIndexer()


def chat_with_rag(user_message: str, history: list) -> str:
    """
    Process user message through RAG retrieval and return answer.

    Args:
        user_message: The user's question
        history: Chat history (for context, if needed)

    Returns:
        Response grounded in SEC + tweet data
    """
    if not user_message.strip():
        return "Please ask a question about the insider trades or social sentiment."

    # Retrieve from both collections
    sec_col = indexer.client.get_or_create_collection("sec_insider_trades")
    tweet_col = indexer.client.get_or_create_collection("tweets")

    sec_results = sec_col.query(query_texts=[user_message], n_results=3)
    tweet_results = tweet_col.query(query_texts=[user_message], n_results=3)

    sec_docs = sec_results['documents'][0] if sec_results['documents'] else []
    tweet_docs = tweet_results['documents'][0] if tweet_results['documents'] else []

    sec_metadata = sec_results['metadatas'][0] if sec_results['metadatas'] else []
    tweet_metadata = tweet_results['metadatas'][0] if tweet_results['metadatas'] else []

    # Build response (plain text only)
    response_parts = []

    if sec_docs:
        response_parts.append("SEC INSIDER TRADES:")
        for i, (doc, meta) in enumerate(zip(sec_docs, sec_metadata), 1):
            ticker = meta.get('ticker', 'N/A')
            value = meta.get('total_value', 0)
            value_str = "${:,.0f}".format(value) if value else "unknown"
            response_parts.append("")
            response_parts.append("{}. {} ({})".format(i, ticker, value_str))
            response_parts.append("   {}".format(doc[:180]))

    if tweet_docs:
        response_parts.append("")
        response_parts.append("SOCIAL SENTIMENT (X/TWITTER):")
        for i, (doc, meta) in enumerate(zip(tweet_docs, tweet_metadata), 1):
            author = meta.get('author', 'unknown')
            likes = meta.get('likes', 0)
            response_parts.append("")
            response_parts.append("{}. @{} ({} likes)".format(i, author, likes))
            response_parts.append("   {}".format(doc[:180]))

    if not sec_docs and not tweet_docs:
        response_parts.append("No matching data found. Try asking about CRWV, NTRA, or SYRE.")

    return "\n".join(response_parts)


def build_ui():
    """Build and return the Gradio interface."""
    with gr.Blocks(title="CrowdWisdomTrading RAG Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # CrowdWisdomTrading Insider Intelligence Chatbot

        Chat with live SEC Form 4 insider trading data + social sentiment from X/Twitter.

        **Data Status:**
        - SEC Trades: 5 live form 4 filings (last 24h)
        - Tweets: 227 real posts from X (7 days)
        - Index: 213 semantic chunks in ChromaDB
        """)

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400,
                    show_label=True,
                )

        with gr.Row():
            with gr.Column(scale=9):
                msg = gr.Textbox(
                    label="Ask about insider trades or social sentiment",
                    placeholder="e.g., What insider trades happened in CRWV? Who is talking about NTRA?",
                    lines=2,
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Send", size="lg")

        # Example questions
        gr.Examples(
            examples=[
                "What major insider trades happened in CRWV?",
                "Who are the top traders discussing these stocks?",
                "What do social media users think about NTRA?",
                "Show me recent activity in SYRE",
                "Which insider sold the most shares?",
            ],
            inputs=msg,
            label="Example Questions",
        )

        # Chat logic
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history

            bot_message = chat_with_rag(message, chat_history)
            # Gradio Chatbot expects (user_message, bot_message) tuples
            chat_history.append((message, bot_message))
            return "", chat_history

        submit_btn.click(respond, [msg, chatbot], [msg, chatbot], queue=False)
        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting CrowdWisdomTrading RAG Chatbot...")
    print("=" * 70)
    print("\nOpen your browser to: http://localhost:7860")
    print("=" * 70 + "\n")

    demo = build_ui()
    demo.launch(share=False)
