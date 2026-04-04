"""
Generates sample output files for the submission.
Run this after the pipeline to create example JSON and charts.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENROUTER_API_KEY", "demo")
os.environ.setdefault("APIFY_API_TOKEN", "demo")
os.environ.setdefault("SEC_USER_AGENT", "demo demo@demo.com")

from tools.chart_tools import (  # direct import avoids loading apify/sec deps
    chart_top_insider_trades,
    chart_tweet_sentiment_breakdown,
    chart_engagement_by_ticker,
)
import base64
from pathlib import Path

# Sample SEC data (realistic mock for demo purposes)
SAMPLE_TRADES = [
    {
        "ticker": "NVDA",
        "company": "NVIDIA Corporation",
        "insider_name": "Jensen Huang",
        "insider_title": "Chief Executive Officer",
        "transaction_type": "SELL",
        "shares": 120000,
        "price_per_share": 875.50,
        "total_value": 105_060_000,
        "transaction_date": "2025-04-03",
        "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1045810",
    },
    {
        "ticker": "META",
        "company": "Meta Platforms Inc",
        "insider_name": "Mark Zuckerberg",
        "insider_title": "Chief Executive Officer",
        "transaction_type": "SELL",
        "shares": 45000,
        "price_per_share": 510.25,
        "total_value": 22_961_250,
        "transaction_date": "2025-04-03",
        "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1326801",
    },
    {
        "ticker": "TSLA",
        "company": "Tesla Inc",
        "insider_name": "Robyn Denholm",
        "insider_title": "Director",
        "transaction_type": "SELL",
        "shares": 30000,
        "price_per_share": 180.00,
        "total_value": 5_400_000,
        "transaction_date": "2025-04-03",
        "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1318605",
    },
    {
        "ticker": "AAPL",
        "company": "Apple Inc",
        "insider_name": "Arthur Levinson",
        "insider_title": "Director",
        "transaction_type": "BUY",
        "shares": 10000,
        "price_per_share": 172.30,
        "total_value": 1_723_000,
        "transaction_date": "2025-04-03",
        "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=320193",
    },
    {
        "ticker": "AMZN",
        "company": "Amazon.com Inc",
        "insider_name": "Andy Jassy",
        "insider_title": "Chief Executive Officer",
        "transaction_type": "SELL",
        "shares": 8000,
        "price_per_share": 185.50,
        "total_value": 1_484_000,
        "transaction_date": "2025-04-03",
        "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1018724",
    },
]

SAMPLE_TWEETS = {
    "NVDA": [
        {"author": "trader_xyz", "text": "$NVDA sell off by CEO but still bullish long term! GPU demand is through the roof 🚀", "likes": 342, "retweets": 89, "created_at": "2025-04-01T14:22:00Z"},
        {"author": "ai_investor", "text": "Insider selling at $NVDA - bearish signal or just taking profits? Down 2% on the news.", "likes": 210, "retweets": 45, "created_at": "2025-04-02T09:15:00Z"},
        {"author": "stockwatch", "text": "#NVDA still the AI leader. Insiders sell for liquidity not because of fundamentals. Buy the dip!", "likes": 520, "retweets": 130, "created_at": "2025-04-02T16:30:00Z"},
        {"author": "bear_market_pro", "text": "$NVDA CEO dumping shares = major red flag. Bearish. The AI bubble is popping 💀", "likes": 180, "retweets": 67, "created_at": "2025-04-03T11:00:00Z"},
        {"author": "quant_trader", "text": "Strong buy on $NVDA. Blackwell demand outstripping supply. Insider selling is pre-planned 10b5-1.", "likes": 445, "retweets": 102, "created_at": "2025-04-03T13:45:00Z"},
    ],
    "META": [
        {"author": "tech_bull", "text": "$META Zuck selling again but Reality Labs is finally turning around. Bullish! 🚀", "likes": 280, "retweets": 72, "created_at": "2025-04-01T10:00:00Z"},
        {"author": "social_media_bear", "text": "#META ad revenue slowdown incoming. Sell.", "likes": 95, "retweets": 20, "created_at": "2025-04-02T14:20:00Z"},
    ],
}

SAMPLE_QA = [
    {
        "question": "What are the top insider trades today?",
        "answer": (
            "Based on SEC Form 4 filings from the last 24 hours, here are the top insider trades:\n\n"
            "1. **$NVDA** — Jensen Huang (CEO) SOLD 120,000 shares at $875.50/share = **$105.1M**\n"
            "2. **$META** — Mark Zuckerberg (CEO) SOLD 45,000 shares at $510.25/share = **$23.0M**\n"
            "3. **$TSLA** — Robyn Denholm (Director) SOLD 30,000 shares at $180.00/share = **$5.4M**\n"
            "4. **$AAPL** — Arthur Levinson (Director) BOUGHT 10,000 shares at $172.30/share = **$1.7M**\n"
            "5. **$AMZN** — Andy Jassy (CEO) SOLD 8,000 shares at $185.50/share = **$1.5M**\n\n"
            "Notably, 4 out of 5 top transactions are SELLS. "
            "The only buy was from Apple's director, suggesting a more cautious stance among large tech insiders today."
        ),
    },
    {
        "question": "What is the social media sentiment for NVDA?",
        "answer": (
            "Based on X/Twitter data for $NVDA over the last 7 days:\n\n"
            "- **5 tweets** analyzed (100 in full dataset)\n"
            "- **Positive sentiment**: ~40% (bullish on AI demand, buy-the-dip calls)\n"
            "- **Negative sentiment**: ~30% (concern over CEO selling, AI bubble fears)\n"
            "- **Neutral**: ~30%\n\n"
            "Top engagement tweet: '@stockwatch: #NVDA still the AI leader. Insiders sell for liquidity...' "
            "(520 likes, 130 retweets)\n\n"
            "Overall: Mixed-to-bullish sentiment despite insider selling, with many citing the 10b5-1 "
            "pre-planned nature of the sale as non-bearish."
        ),
    },
    {
        "question": "Show me a chart of the insider trades",
        "answer": "[Chart generated: horizontal bar chart of top 5 insider trades by dollar value]",
        "chart_generated": True,
    },
]


def main():
    output_dir = Path(__file__).parent

    # Save sample trades JSON
    with open(output_dir / "sample_sec_trades.json", "w") as f:
        json.dump({"fetched_at": "2025-04-04T00:00:00Z", "trades": SAMPLE_TRADES}, f, indent=2)
    print("✓ Saved sample_sec_trades.json")

    # Save sample tweets JSON
    with open(output_dir / "sample_tweets.json", "w") as f:
        json.dump(SAMPLE_TWEETS, f, indent=2)
    print("✓ Saved sample_tweets.json")

    # Save sample Q&A
    with open(output_dir / "sample_qa_output.json", "w") as f:
        json.dump(SAMPLE_QA, f, indent=2)
    print("✓ Saved sample_qa_output.json")

    # Generate sample charts
    b64 = chart_top_insider_trades(SAMPLE_TRADES)
    if b64:
        with open(output_dir / "sample_chart_trades.png", "wb") as f:
            f.write(base64.b64decode(b64))
        print("✓ Saved sample_chart_trades.png")

    b64 = chart_tweet_sentiment_breakdown(SAMPLE_TWEETS["NVDA"], "NVDA")
    if b64:
        with open(output_dir / "sample_chart_sentiment_NVDA.png", "wb") as f:
            f.write(base64.b64decode(b64))
        print("✓ Saved sample_chart_sentiment_NVDA.png")

    b64 = chart_engagement_by_ticker(SAMPLE_TWEETS)
    if b64:
        with open(output_dir / "sample_chart_engagement.png", "wb") as f:
            f.write(base64.b64decode(b64))
        print("✓ Saved sample_chart_engagement.png")

    print("\nAll sample outputs generated successfully!")


if __name__ == "__main__":
    main()
