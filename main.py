"""
CrowdWisdomTrading AI SEC Chat
──────────────────────────────
Gradio chatbot backed by:
  - SEC Form 4 insider trading data (last 24h)
  - X/Twitter sentiment for top tickers (last 7d)
  - RAG (ChromaDB + sentence-transformers)
  - Hermes-style function calling agent (OpenRouter)
  - Closed learning loop (feedback → memory → re-ranking)
"""
import base64
import json
import logging
import os
import uuid
from datetime import datetime

import gradio as gr
from rich.logging import RichHandler

from config import PORT
from agents.hermes_agent import HermesAgent, tool
from rag.retriever import RAGRetriever
from feedback.learning_loop import LearningLoop
from tools.chart_tools import (
    chart_top_insider_trades,
    chart_tweet_volume_over_time,
    chart_tweet_sentiment_breakdown,
    chart_insider_trade_timeline,
    chart_engagement_by_ticker,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
retriever = RAGRetriever()
learning_loop = LearningLoop()

# ── Tool definitions ──────────────────────────────────────────────────────────

@tool(
    name="search_insider_trades",
    description="Search SEC insider trading data for a specific ticker or insider name.",
    parameters={
        "query": {"type": "string", "description": "Search query (ticker symbol or insider name)", "required": True},
        "ticker": {"type": "string", "description": "Optional: filter by specific ticker symbol"},
    },
)
def search_insider_trades(query: str, ticker: str = "") -> str:
    results = retriever.retrieve(
        query, filter_ticker=ticker or None, data_types=["sec_trade"]
    )
    return retriever.format_context(results)


@tool(
    name="search_tweets",
    description="Search X/Twitter sentiment data for a ticker or topic.",
    parameters={
        "query": {"type": "string", "description": "Search query about market sentiment", "required": True},
        "ticker": {"type": "string", "description": "Optional: filter by specific ticker symbol"},
    },
)
def search_tweets(query: str, ticker: str = "") -> str:
    results = retriever.retrieve(
        query, filter_ticker=ticker or None, data_types=["tweet"]
    )
    return retriever.format_context(results)


@tool(
    name="get_ticker_summary",
    description="Get a summary of all available data for a specific stock ticker.",
    parameters={
        "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)", "required": True},
    },
)
def get_ticker_summary(ticker: str) -> str:
    summary = retriever.get_ticker_summary(ticker.upper())
    trades = summary.get("sec_trades", [])
    return json.dumps({
        "ticker": ticker.upper(),
        "insider_trade_count": len(trades),
        "total_value_traded": f"${summary.get('total_value', 0):,.0f}",
        "tweet_count": summary.get("tweet_count", 0),
        "trades": trades[:3],  # First 3 trades as preview
    }, indent=2)


@tool(
    name="get_all_tickers",
    description="List all stock tickers available in the knowledge base.",
    parameters={},
)
def get_all_tickers() -> str:
    tickers = retriever.get_all_tickers()
    return f"Available tickers: {', '.join(tickers) if tickers else 'No data loaded yet. Run the pipeline first.'}"


@tool(
    name="chart_insider_trades_bar",
    description="Generate a bar chart of top SEC insider trades by dollar value. Call this when user asks for a chart of trades.",
    parameters={},
)
def chart_insider_trades_bar() -> str:
    """Returns base64 PNG chart."""
    from config import SEC_DATA_DIR
    from pathlib import Path
    path = Path(SEC_DATA_DIR) / "latest_insider_trades.json"
    if not path.exists():
        return "No SEC data available. Run the pipeline first."
    with open(path) as f:
        data = json.load(f)
    trades = data.get("trades", [])
    return chart_top_insider_trades(trades)


@tool(
    name="chart_tweet_volume",
    description="Generate a line chart of tweet volume over time for a ticker.",
    parameters={
        "ticker": {"type": "string", "description": "Stock ticker symbol", "required": True},
    },
)
def chart_tweet_volume(ticker: str) -> str:
    from config import TWEETS_DIR
    from pathlib import Path
    path = Path(TWEETS_DIR) / f"{ticker.upper()}_tweets.json"
    if not path.exists():
        return f"No tweet data for {ticker}."
    with open(path) as f:
        data = json.load(f)
    tweets = data.get("tweets", [])
    return chart_tweet_volume_over_time(tweets, ticker.upper())


@tool(
    name="chart_sentiment_donut",
    description="Generate a sentiment donut chart (positive/negative/neutral) for a ticker's tweets.",
    parameters={
        "ticker": {"type": "string", "description": "Stock ticker symbol", "required": True},
    },
)
def chart_sentiment_donut(ticker: str) -> str:
    from config import TWEETS_DIR
    from pathlib import Path
    path = Path(TWEETS_DIR) / f"{ticker.upper()}_tweets.json"
    if not path.exists():
        return f"No tweet data for {ticker}."
    with open(path) as f:
        data = json.load(f)
    tweets = data.get("tweets", [])
    return chart_tweet_sentiment_breakdown(tweets, ticker.upper())


@tool(
    name="chart_engagement_comparison",
    description="Generate a bar chart comparing social media engagement across all tickers.",
    parameters={},
)
def chart_engagement_comparison() -> str:
    from config import TWEETS_DIR
    from pathlib import Path
    tweet_data = {}
    for p in Path(TWEETS_DIR).glob("*_tweets.json"):
        with open(p) as f:
            data = json.load(f)
        ticker = data.get("ticker", p.stem.replace("_tweets", ""))
        tweet_data[ticker] = data.get("tweets", [])
    return chart_engagement_by_ticker(tweet_data)


# ── Agent setup ───────────────────────────────────────────────────────────────

TOOLS_MAP = {
    "search_insider_trades": search_insider_trades,
    "search_tweets": search_tweets,
    "get_ticker_summary": get_ticker_summary,
    "get_all_tickers": get_all_tickers,
    "chart_insider_trades_bar": chart_insider_trades_bar,
    "chart_tweet_volume": chart_tweet_volume,
    "chart_sentiment_donut": chart_sentiment_donut,
    "chart_engagement_comparison": chart_engagement_comparison,
}

agent = HermesAgent(tools=TOOLS_MAP, max_iterations=5)

# ── Gradio chat logic ─────────────────────────────────────────────────────────

SESSION_ID = str(uuid.uuid4())
interaction_ids = []  # Track interaction IDs for feedback


def respond(message: str, history: list[list]) -> tuple:
    """
    Process a user message and return (updated_history, image_or_none).
    """
    if not message.strip():
        return history, None

    # Build conversation history for the agent
    conv_history = []
    for user_msg, bot_msg in history[-3:]:  # Last 3 turns
        if user_msg:
            conv_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            # Strip image markdown for history
            clean = bot_msg.split("![chart]")[0].strip() if "![chart]" in bot_msg else bot_msg
            conv_history.append({"role": "assistant", "content": clean})

    # Retrieve context
    rag_results = retriever.retrieve(message)
    rag_results = learning_loop.rerank_results(rag_results)
    context = retriever.format_context(rag_results)
    few_shots = learning_loop.get_few_shot_examples(n=2)

    # Run the Hermes agent
    try:
        answer, charts = agent.chat(
            user_message=message,
            context=context,
            conversation_history=conv_history,
            few_shot_examples=few_shots,
            session_id=SESSION_ID,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        answer = f"An error occurred: {str(e)}"
        charts = []

    # Log interaction for learning loop
    int_id = learning_loop.log_interaction(
        query=message,
        answer=answer,
        context_used=rag_results,
        session_id=SESSION_ID,
    )
    interaction_ids.append(int_id)

    # Handle chart images
    chart_image = None
    if charts:
        # Return the last chart (most relevant)
        chart_image = base64.b64decode(charts[-1])

    history.append([message, answer])
    return history, chart_image


def give_feedback(rating: int, history: list[list]) -> str:
    """Record feedback for the last interaction."""
    if not interaction_ids:
        return "No interactions to rate yet."
    last_id = interaction_ids[-1]
    learning_loop.record_feedback(last_id, rating=rating)
    emoji = "👍" if rating == 1 else "👎"
    return f"{emoji} Feedback recorded! This helps improve future answers."


def get_learning_stats() -> str:
    stats = learning_loop.get_stats()
    lines = [
        "## Learning Loop Stats",
        f"- Total interactions: {stats['total_interactions']}",
        f"- Rated interactions: {stats['rated']}",
        f"- Positive ratings: {stats['positive_ratings']}",
        f"- Negative ratings: {stats['negative_ratings']}",
        f"- Satisfaction rate: {stats['satisfaction_rate']}",
        f"- Examples in memory: {stats['successful_examples_in_memory']}",
    ]
    if stats.get("ticker_weights"):
        lines.append("\n**Ticker Weights (learned from feedback):**")
        for ticker, weight in stats["ticker_weights"].items():
            lines.append(f"  - {ticker}: {weight:.2f}")
    return "\n".join(lines)


def run_pipeline_ui(refresh: bool) -> str:
    """Run the data pipeline from the UI."""
    try:
        from flow.pipeline import DataPipeline
        pipeline = DataPipeline()
        result = pipeline.run(refresh_data=refresh)
        trades = result.get("trades", [])
        tickers = result.get("tickers", [])
        total_tweets = sum(len(v) for v in result.get("tweets", {}).values())
        return (
            f"Pipeline complete!\n"
            f"Tickers: {', '.join(tickers)}\n"
            f"Trades indexed: {len(trades)}\n"
            f"Tweets indexed: {total_tweets}\n"
            f"DB stats: {result.get('stats', {})}"
        )
    except Exception as e:
        return f"Pipeline error: {e}"


# ── Gradio UI ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "What are the top insider trades in the last 24 hours?",
    "Show me a chart of insider trading activity",
    "What is the social media sentiment for the top traded tickers?",
    "Which insiders made the biggest buys today?",
    "Show tweet volume chart for [TICKER]",
    "Compare social engagement across all tickers",
    "What are people saying about [TICKER] on X?",
    "Show me a sentiment donut chart for [TICKER]",
]


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="CrowdWisdomTrading AI SEC Chat") as demo:
        gr.Markdown("""
# 📈 CrowdWisdomTrading AI SEC Chat
**Real-time SEC insider trading intelligence + X/Twitter sentiment**

> Powered by Hermes Agent (OpenRouter) • RAG (ChromaDB) • APIFY
        """)

        with gr.Tabs():
            # ── Chat tab ─────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="SEC Intelligence Chat",
                            height=500,
                            show_copy_button=True,
                        )
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Ask about insider trades, sentiment, charts...",
                                label="Your question",
                                scale=4,
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1)

                        with gr.Row():
                            thumbs_up = gr.Button("👍 Helpful", variant="secondary")
                            thumbs_down = gr.Button("👎 Not helpful", variant="stop")
                            feedback_msg = gr.Textbox(label="Feedback status", interactive=False, scale=3)

                    with gr.Column(scale=2):
                        chart_output = gr.Image(
                            label="Chart Output",
                            type="filepath",
                            interactive=False,
                        )
                        gr.Markdown("### 💡 Sample Questions")
                        for q in SAMPLE_QUESTIONS:
                            gr.Markdown(f"- {q}")

            # ── Pipeline tab ─────────────────────────────────────────────
            with gr.Tab("⚙️ Data Pipeline"):
                gr.Markdown("""
## Data Collection Pipeline
Fetches latest SEC Form 4 filings and X/Twitter data.
Run this before chatting to load fresh data.
                """)
                with gr.Row():
                    refresh_check = gr.Checkbox(label="Fetch fresh data (uncheck to use cache)", value=True)
                    run_pipeline_btn = gr.Button("🚀 Run Pipeline", variant="primary")
                pipeline_output = gr.Textbox(label="Pipeline output", lines=10, interactive=False)

            # ── Learning Loop tab ─────────────────────────────────────────
            with gr.Tab("🧠 Learning Loop"):
                gr.Markdown("""
## Closed Learning Loop Stats
The agent learns from your feedback over time.
Positive feedback → stores examples in memory, boosts ticker weights.
Negative feedback → logs failure patterns, adjusts retrieval.
                """)
                stats_btn = gr.Button("Refresh Stats")
                stats_output = gr.Markdown()

        # ── Event handlers ────────────────────────────────────────────────

        def on_submit(message, history):
            if not message.strip():
                return history, None, ""
            updated_history, chart_img_bytes = respond(message, history)
            chart_path = None
            if chart_img_bytes:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(chart_img_bytes)
                    chart_path = tmp.name
            return updated_history, chart_path, ""

        submit_btn.click(
            on_submit, [msg, chatbot], [chatbot, chart_output, msg]
        )
        msg.submit(
            on_submit, [msg, chatbot], [chatbot, chart_output, msg]
        )

        thumbs_up.click(
            lambda h: give_feedback(1, h), [chatbot], [feedback_msg]
        )
        thumbs_down.click(
            lambda h: give_feedback(-1, h), [chatbot], [feedback_msg]
        )

        run_pipeline_btn.click(
            run_pipeline_ui, [refresh_check], [pipeline_output]
        )

        stats_btn.click(lambda: get_learning_stats(), [], [stats_output])
        demo.load(lambda: get_learning_stats(), [], [stats_output])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_port=PORT,
        share=False,
        show_error=True,
    )
