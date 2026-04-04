"""
Chart generation tool for insider trading and tweet sentiment visualization.
Returns base64-encoded PNG images for embedding in the chat UI.
"""
import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from config import CHARTS_DIR

logger = logging.getLogger(__name__)

COLORS = {
    "BUY": "#2ecc71",
    "SELL": "#e74c3c",
    "neutral": "#3498db",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "bg": "#1a1a2e",
    "grid": "#2d2d4e",
    "text": "#e0e0e0",
}


def _apply_dark_theme(fig, ax):
    """Apply a clean dark theme to the chart."""
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], linestyle="--", alpha=0.5)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=COLORS["bg"])
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def chart_top_insider_trades(trades: list[dict]) -> str:
    """
    Horizontal bar chart of top insider trades by dollar value.
    Returns base64 PNG.
    """
    if not trades:
        return ""

    df = pd.DataFrame(trades)
    df["label"] = df["ticker"] + "\n" + df["insider_name"].str[:20]
    df["value_m"] = df["total_value"] / 1_000_000
    df = df.sort_values("total_value")

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    bar_colors = [COLORS["BUY"] if t == "BUY" else COLORS["SELL"] for t in df["transaction_type"]]
    bars = ax.barh(df["label"], df["value_m"], color=bar_colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, df["value_m"]):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"${val:.2f}M", va="center", ha="left", color=COLORS["text"], fontsize=9
        )

    ax.set_xlabel("Transaction Value ($ Millions)", color=COLORS["text"])
    ax.set_title("Top SEC Insider Trades (Last 24h)", color=COLORS["text"], fontsize=14, pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["BUY"], label="Buy"),
        Patch(facecolor=COLORS["SELL"], label="Sell"),
    ]
    ax.legend(handles=legend_elements, facecolor=COLORS["bg"], labelcolor=COLORS["text"])

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    _save_chart(b64, "top_insider_trades")
    return b64


def chart_tweet_volume_over_time(tweets: list[dict], ticker: str) -> str:
    """
    Line chart of tweet volume per day for a ticker.
    Returns base64 PNG.
    """
    if not tweets:
        return ""

    dates = []
    for t in tweets:
        raw = t.get("created_at", "")
        try:
            dt = pd.to_datetime(raw, utc=True).date()
            dates.append(dt)
        except Exception:
            continue

    if not dates:
        return ""

    series = pd.Series(dates).value_counts().sort_index()
    series.index = pd.to_datetime(series.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    ax.plot(series.index, series.values, color=COLORS["neutral"], linewidth=2, marker="o", markersize=5)
    ax.fill_between(series.index, series.values, alpha=0.2, color=COLORS["neutral"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    ax.set_ylabel("Tweet Count", color=COLORS["text"])
    ax.set_title(f"${ticker} Tweet Volume (Last 7 Days)", color=COLORS["text"], fontsize=14, pad=15)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    _save_chart(b64, f"{ticker}_tweet_volume")
    return b64


def chart_tweet_sentiment_breakdown(tweets: list[dict], ticker: str) -> str:
    """
    Donut chart of positive/negative/neutral tweet sentiment (keyword-based).
    Returns base64 PNG.
    """
    if not tweets:
        return ""

    positive_kw = {"bullish", "buy", "moon", "up", "long", "surge", "beat", "great",
                   "strong", "rally", "growth", "profit", "gain", "outperform", "🚀"}
    negative_kw = {"bearish", "sell", "down", "short", "crash", "miss", "weak",
                   "drop", "loss", "fail", "decline", "underperform", "💀", "🐻"}

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for t in tweets:
        text_lower = t.get("text", "").lower()
        words = set(text_lower.split())
        if words & positive_kw:
            counts["Positive"] += 1
        elif words & negative_kw:
            counts["Negative"] += 1
        else:
            counts["Neutral"] += 1

    labels = list(counts.keys())
    sizes = list(counts.values())
    colors_list = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]

    fig, ax = plt.subplots(figsize=(7, 7))
    _apply_dark_theme(fig, ax)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_list, autopct="%1.1f%%",
        startangle=140, wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
        pctdistance=0.75,
    )
    for text in texts + autotexts:
        text.set_color(COLORS["text"])
    ax.set_title(f"${ticker} Tweet Sentiment", color=COLORS["text"], fontsize=14, pad=20)

    # Draw donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc=COLORS["bg"])
    ax.add_artist(centre_circle)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    _save_chart(b64, f"{ticker}_sentiment")
    return b64


def chart_insider_trade_timeline(trades: list[dict]) -> str:
    """
    Scatter plot of all trades over time with size proportional to value.
    Returns base64 PNG.
    """
    if not trades:
        return ""

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_theme(fig, ax)

    for _, row in df.iterrows():
        color = COLORS["BUY"] if row["transaction_type"] == "BUY" else COLORS["SELL"]
        size = max(50, min(1000, row["total_value"] / 10_000))
        ax.scatter(row["date"], row["total_value"] / 1_000_000,
                   s=size, color=color, alpha=0.8, edgecolors="none")
        ax.annotate(row["ticker"], (row["date"], row["total_value"] / 1_000_000),
                    fontsize=8, color=COLORS["text"], ha="center", va="bottom")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    ax.set_ylabel("Trade Value ($M)", color=COLORS["text"])
    ax.set_title("Insider Trade Timeline", color=COLORS["text"], fontsize=14, pad=15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["BUY"], label="Buy"),
        Patch(facecolor=COLORS["SELL"], label="Sell"),
    ]
    ax.legend(handles=legend_elements, facecolor=COLORS["bg"], labelcolor=COLORS["text"])

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    _save_chart(b64, "insider_timeline")
    return b64


def chart_engagement_by_ticker(tweet_data: dict[str, list[dict]]) -> str:
    """
    Grouped bar chart: likes + retweets per ticker.
    Returns base64 PNG.
    """
    if not tweet_data:
        return ""

    tickers, likes_list, rt_list = [], [], []
    for ticker, tweets in tweet_data.items():
        if tweets:
            tickers.append(ticker)
            likes_list.append(sum(t.get("likes", 0) for t in tweets))
            rt_list.append(sum(t.get("retweets", 0) for t in tweets))

    if not tickers:
        return ""

    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    ax.bar(x - width / 2, likes_list, width, label="Likes", color=COLORS["positive"], alpha=0.9)
    ax.bar(x + width / 2, rt_list, width, label="Retweets", color=COLORS["neutral"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"${t}" for t in tickers], color=COLORS["text"])
    ax.set_ylabel("Count", color=COLORS["text"])
    ax.set_title("Social Engagement by Ticker (Last 7 Days)", color=COLORS["text"], fontsize=14, pad=15)
    ax.legend(facecolor=COLORS["bg"], labelcolor=COLORS["text"])

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    _save_chart(b64, "engagement_by_ticker")
    return b64


def _save_chart(b64: str, name: str):
    """Save chart PNG to disk for reference."""
    path = Path(CHARTS_DIR) / f"{name}.png"
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    logger.debug(f"Saved chart to {path}")
