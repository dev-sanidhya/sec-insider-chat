__all__ = [
    "SECInsiderTradingFetcher",
    "TwitterScraper",
    "chart_top_insider_trades",
    "chart_tweet_volume_over_time",
    "chart_tweet_sentiment_breakdown",
    "chart_insider_trade_timeline",
    "chart_engagement_by_ticker",
]


def __getattr__(name):
    if name == "SECInsiderTradingFetcher":
        from tools.sec_tools import SECInsiderTradingFetcher
        return SECInsiderTradingFetcher
    if name == "TwitterScraper":
        from tools.apify_tools import TwitterScraper
        return TwitterScraper
    if name in ("chart_top_insider_trades", "chart_tweet_volume_over_time",
                "chart_tweet_sentiment_breakdown", "chart_insider_trade_timeline",
                "chart_engagement_by_ticker"):
        from tools import chart_tools
        return getattr(chart_tools, name)
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
