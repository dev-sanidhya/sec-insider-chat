"""
SEC EDGAR Form 4 insider trading data fetcher + signal filters.

Integrates patterns from:
  - wescules/insider-trading-analyzer : cluster_buys, big_money, repeated_buyer filters
  - TradeSignal                        : BULLISH/BEARISH/NEUTRAL signal classification
  - sec-edgar-agentkit                 : optional smolagents InsiderTradingTool wrapper
  - sec-api-python                     : optional premium data source (needs paid key)
"""
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from tenacity import retry, stop_after_attempt, wait_exponential

from config import SEC_EDGAR_BASE, SEC_FORM4_RSS, SEC_USER_AGENT, SEC_DATA_DIR

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
DB_PATH = Path(SEC_DATA_DIR) / "insider_trades.db"


# ─────────────────────────────────────────────────────────────────────────────
# SQLite persistence  (pattern from insider-trading-analyzer)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_database() -> sqlite3.Connection:
    """Create SQLite schema matching insider-trading-analyzer structure."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS insider_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            company TEXT,
            insider_name TEXT,
            insider_title TEXT,
            transaction_type TEXT,   -- 'BUY' or 'SELL'
            transaction_code TEXT,   -- raw SEC code: P, S, etc.
            shares REAL,
            price_per_share REAL,
            total_value REAL,
            transaction_date TEXT,
            is_10b5_plan INTEGER DEFAULT 0,
            source_url TEXT,
            filing_updated TEXT,
            fetched_at TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON insider_trades(ticker)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date ON insider_trades(transaction_date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_insider ON insider_trades(insider_name)")
    con.commit()
    return con


def insert_trades(con: sqlite3.Connection, trades: list[dict]):
    """Upsert trades into SQLite, avoiding duplicates."""
    now = datetime.utcnow().isoformat()
    for t in trades:
        existing = con.execute(
            "SELECT id FROM insider_trades WHERE ticker=? AND insider_name=? AND transaction_date=? AND shares=?",
            (t.get("ticker"), t.get("insider_name"), t.get("transaction_date"), t.get("shares")),
        ).fetchone()
        if not existing:
            con.execute(
                """INSERT INTO insider_trades
                   (ticker, company, insider_name, insider_title, transaction_type, transaction_code,
                    shares, price_per_share, total_value, transaction_date, is_10b5_plan,
                    source_url, filing_updated, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    t.get("ticker", ""), t.get("company", ""), t.get("insider_name", ""),
                    t.get("insider_title", ""), t.get("transaction_type", ""),
                    t.get("transaction_code", ""), t.get("shares", 0), t.get("price_per_share", 0),
                    t.get("total_value", 0), t.get("transaction_date", ""),
                    int(t.get("is_10b5_plan", False)), t.get("source_url", ""),
                    t.get("filing_updated", ""), now,
                ),
            )
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Signal filters  (adapted from wescules/insider-trading-analyzer filter.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_cluster_buys(
    con: sqlite3.Connection,
    min_insiders: int = 2,
    window_days: int = 5,
    min_value: float = 10_000,
) -> list[dict]:
    """
    Coordinated cluster buys: multiple insiders buying the same ticker
    within a short window. Excludes 10b5-1 pre-planned trades.
    Adapted from insider-trading-analyzer cluster_buys_query().
    """
    cutoff = (datetime.utcnow() - timedelta(days=window_days)).strftime("%Y-%m-%d")
    rows = con.execute(
        """
        SELECT ticker, transaction_date, COUNT(DISTINCT insider_name) AS insider_count,
               SUM(total_value) AS cluster_value,
               GROUP_CONCAT(insider_name, ' | ') AS insiders
        FROM insider_trades
        WHERE transaction_type = 'BUY'
          AND is_10b5_plan = 0
          AND total_value >= ?
          AND transaction_date >= ?
        GROUP BY ticker, DATE(transaction_date)
        HAVING insider_count >= ?
        ORDER BY cluster_value DESC
        """,
        (min_value, cutoff, min_insiders),
    ).fetchall()
    return [dict(r) for r in rows]


def find_big_money_trades(
    con: sqlite3.Connection,
    min_value: float = 500_000,
    transaction_type: str = "BUY",
    days: int = 7,
) -> list[dict]:
    """
    Single large transactions above a dollar threshold.
    Adapted from insider-trading-analyzer big_money_query().
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = con.execute(
        """
        SELECT ticker, company, insider_name, insider_title, transaction_type,
               shares, price_per_share, total_value, transaction_date, source_url
        FROM insider_trades
        WHERE transaction_type = ?
          AND total_value >= ?
          AND transaction_date >= ?
        ORDER BY total_value DESC
        """,
        (transaction_type, min_value, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]


def find_repeated_buyers(
    con: sqlite3.Connection,
    min_buys: int = 2,
    days: int = 30,
) -> list[dict]:
    """
    Insiders who bought the same ticker multiple times — strong conviction signal.
    Adapted from insider-trading-analyzer repeated_buyer_query().
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = con.execute(
        """
        SELECT ticker, insider_name, insider_title,
               COUNT(*) AS buy_count,
               SUM(total_value) AS total_invested,
               MIN(transaction_date) AS first_buy,
               MAX(transaction_date) AS last_buy
        FROM insider_trades
        WHERE transaction_type = 'BUY'
          AND transaction_date >= ?
        GROUP BY ticker, insider_name
        HAVING buy_count >= ?
        ORDER BY total_invested DESC
        """,
        (cutoff, min_buys),
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Signal classification  (inspired by TradeSignal's AI signals endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def classify_signal(trades: list[dict]) -> dict:
    """
    Classify a list of trades as BULLISH / BEARISH / NEUTRAL with a confidence score.
    Logic inspired by TradeSignal's BULLISH/BEARISH/NEUTRAL classification.

    Factors:
      - Buy/Sell ratio by dollar value
      - Cluster buy detection (multiple insiders = stronger signal)
      - Insider title (CEO/CFO weight more than Director)
    """
    if not trades:
        return {"signal": "NEUTRAL", "confidence": 0.0, "reason": "No trades to analyze"}

    title_weights = {
        "chief executive officer": 2.0, "ceo": 2.0,
        "chief financial officer": 1.8, "cfo": 1.8,
        "chief operating officer": 1.6, "coo": 1.6,
        "president": 1.5,
        "director": 1.2,
        "vp": 1.1, "vice president": 1.1,
    }

    buy_value = 0.0
    sell_value = 0.0
    buy_insiders = set()
    sell_insiders = set()

    for t in trades:
        title = t.get("insider_title", "").lower()
        weight = 1.0
        for key, w in title_weights.items():
            if key in title:
                weight = w
                break

        val = t.get("total_value", 0) * weight
        if t.get("transaction_type") == "BUY":
            buy_value += val
            buy_insiders.add(t.get("insider_name", ""))
        else:
            sell_value += val
            sell_insiders.add(t.get("insider_name", ""))

    total = buy_value + sell_value
    if total == 0:
        return {"signal": "NEUTRAL", "confidence": 0.0, "reason": "Zero trade value"}

    buy_ratio = buy_value / total
    cluster_bonus = 0.1 if len(buy_insiders) >= 2 else 0.0

    if buy_ratio >= 0.65 + cluster_bonus:
        signal = "BULLISH"
        confidence = min(1.0, buy_ratio + cluster_bonus)
        reason = (
            f"{len(buy_insiders)} insider(s) buying ${buy_value:,.0f} vs "
            f"${sell_value:,.0f} sold"
        )
    elif buy_ratio <= 0.35 - cluster_bonus:
        signal = "BEARISH"
        confidence = min(1.0, (1 - buy_ratio) + cluster_bonus * 0.5)
        reason = (
            f"Heavy selling: ${sell_value:,.0f} sold vs ${buy_value:,.0f} bought "
            f"by {len(sell_insiders)} insider(s)"
        )
    else:
        signal = "NEUTRAL"
        confidence = 0.5
        reason = f"Mixed signals: {buy_ratio:.0%} buys by value"

    return {"signal": signal, "confidence": round(confidence, 2), "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# Optional: sec-api-python premium integration
# ─────────────────────────────────────────────────────────────────────────────

def fetch_via_sec_api(ticker: str, days: int = 1) -> list[dict]:
    """
    Fetch insider trades using the sec-api.io commercial API (janlukasschroeder/sec-api-python).
    Requires SEC_API_KEY in environment. Falls back gracefully if not set.
    """
    api_key = os.getenv("SEC_API_KEY", "")
    if not api_key:
        logger.debug("SEC_API_KEY not set — skipping sec-api.io premium source")
        return []

    try:
        from sec_api import InsiderTradingApi
        api = InsiderTradingApi(api_key)
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        data = api.get_data({
            "query": f"issuer.tradingSymbol:{ticker} AND transactionDate:[{since} TO *]",
            "from": "0",
            "size": "50",
            "sort": [{"transactionDate": {"order": "desc"}}],
        })
        trades = []
        for item in data.get("transactions", []):
            val = item.get("amounts", {})
            shares = float(val.get("shares", 0) or 0)
            price = float(val.get("pricePerShare", 0) or 0)
            trades.append({
                "ticker": ticker,
                "company": item.get("issuer", {}).get("name", ""),
                "insider_name": item.get("reportingOwner", {}).get("name", ""),
                "insider_title": item.get("reportingOwner", {}).get("relationship", {}).get("officerTitle", ""),
                "transaction_type": "BUY" if item.get("transactionCode") == "P" else "SELL",
                "transaction_code": item.get("transactionCode", ""),
                "shares": shares,
                "price_per_share": price,
                "total_value": shares * price,
                "transaction_date": item.get("transactionDate", ""),
                "source_url": item.get("filingUrl", ""),
                "is_10b5_plan": False,
            })
        logger.info(f"sec-api.io returned {len(trades)} trades for {ticker}")
        return trades
    except Exception as e:
        logger.warning(f"sec-api.io fetch failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Optional: sec-edgar-agentkit smolagents InsiderTradingTool wrapper
# ─────────────────────────────────────────────────────────────────────────────

def fetch_via_agentkit(query: str) -> str:
    """
    Use the sec-edgar-agentkit InsiderTradingTool (smolagents) to answer
    a natural language question about insider trading.
    Requires: pip install sec-edgar-agentkit-smolagents sec-edgar-mcp
    Falls back gracefully if not installed.
    """
    try:
        from sec_edgar_smolagents import InsiderTradingTool
        tool = InsiderTradingTool()
        result = tool(query)
        return str(result)
    except ImportError:
        logger.debug("sec-edgar-agentkit-smolagents not installed — skipping")
        return ""
    except Exception as e:
        logger.warning(f"sec-edgar-agentkit query failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main fetcher
# ─────────────────────────────────────────────────────────────────────────────

class SECInsiderTradingFetcher:
    """
    Fetches and parses SEC Form 4 insider trading filings from EDGAR.
    Persists to SQLite and applies insider-trading-analyzer filter patterns.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.db = initialize_database()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get(self, url: str, params: dict = None) -> requests.Response:
        time.sleep(0.1)  # SEC rate limit: 10 req/sec
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp

    def fetch_recent_form4_filings(self, hours: int = 24) -> list[dict]:
        """Fetch Form 4 filing metadata from SEC EDGAR RSS feed."""
        logger.info(f"Fetching Form 4 filings from the last {hours} hours...")
        params = {
            "action": "getcurrent",
            "type": "4",
            "dateb": "",
            "owner": "include",
            "count": "100",
            "search_text": "",
            "output": "atom",
        }
        filings = []
        seen_accessions = set()
        try:
            resp = self._get(SEC_FORM4_RSS, params=params)
            # Parse as HTML since BeautifulSoup xml chokes on ISO-8859-1 Atom feeds
            soup = BeautifulSoup(resp.content, "lxml")
            entries = soup.find_all("entry")
            logger.info(f"Found {len(entries)} raw entries in RSS feed")

            for entry in entries:
                try:
                    filing = self._parse_rss_entry(entry)
                    if filing:
                        acc = filing.get("accession", "")
                        if acc and acc not in seen_accessions:
                            seen_accessions.add(acc)
                            filings.append(filing)
                except Exception as e:
                    logger.debug(f"Error parsing entry: {e}")

            logger.info(f"Unique filings after dedup: {len(filings)}")
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {e}")
        return filings

    def _parse_rss_entry(self, entry) -> Optional[dict]:
        """Parse a single Atom entry from the SEC EDGAR RSS feed."""
        # The feed uses <link rel="alternate" href="..."> for the filing URL
        link_tag = entry.find("link", rel="alternate") or entry.find("link")
        href = ""
        if link_tag:
            href = link_tag.get("href", "")

        if not href:
            return None

        title_tag = entry.find("title")
        updated_tag = entry.find("updated")
        id_tag = entry.find("id")

        title = title_tag.text.strip() if title_tag else ""
        updated = updated_tag.text.strip() if updated_tag else ""

        # Extract accession number from the id tag (urn:tag:sec.gov,...:accession-number=...)
        accession = ""
        if id_tag:
            raw_id = id_tag.text.strip()
            if "accession-number=" in raw_id:
                accession = raw_id.split("accession-number=")[-1]

        return {
            "title": title,
            "updated": updated,
            "filing_href": href,
            "accession": accession,
        }

    def parse_form4_transaction(self, filing_href: str) -> list[dict]:
        """Parse a Form 4 filing index page, then fetch and parse the raw XML."""
        transactions = []
        try:
            resp = self._get(filing_href)
            soup = BeautifulSoup(resp.content, "html.parser")
            xml_link = None
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Must be .xml, not the XSL-rendered version (xslF345X06 folder), not XBRL
                if (href.endswith(".xml")
                        and "xsl" not in href.lower()
                        and "xbrl" not in href.lower()):
                    xml_link = f"https://www.sec.gov{href}" if href.startswith("/") else href
                    break
            if not xml_link:
                return transactions
            xml_resp = self._get(xml_link)
            transactions = self._parse_form4_xml(xml_resp.content, filing_href)
        except Exception as e:
            logger.debug(f"Error parsing filing {filing_href}: {e}")
        return transactions

    def _parse_form4_xml(self, xml_content: bytes, source_url: str) -> list[dict]:
        """Parse Form 4 XML and extract non-derivative transactions."""
        transactions = []
        try:
            soup = BeautifulSoup(xml_content, "xml")

            issuer = soup.find("issuer")
            ticker, company = "", ""
            if issuer:
                ticker = (issuer.find("issuerTradingSymbol") or soup.new_tag("x")).text.strip()
                company = (issuer.find("issuerName") or soup.new_tag("x")).text.strip()

            owner = soup.find("reportingOwner")
            insider_name, insider_title = "", ""
            if owner:
                insider_name = (owner.find("rptOwnerName") or soup.new_tag("x")).text.strip()
                insider_title = (owner.find("officerTitle") or soup.new_tag("x")).text.strip()

            # Detect 10b5-1 plan (pre-planned, less significant signal)
            footnotes_text = " ".join(
                f.text.lower() for f in soup.find_all("footnote")
            )
            is_10b5_plan = "10b5-1" in footnotes_text or "10b5" in footnotes_text

            for txn in soup.find_all("nonDerivativeTransaction"):
                try:
                    def val(tag_name):
                        """Extract value from a tag — handles both <tag><value>X</value></tag>
                        and <tag>X</tag> formats found in SEC Form 4 XML."""
                        t = txn.find(tag_name)
                        if not t:
                            return ""
                        v = t.find("value")
                        if v:
                            return v.text.strip()
                        # Direct text (no <value> child)
                        return t.text.strip()

                    date_str = val("transactionDate")
                    code = val("transactionCode")
                    shares_str = val("transactionShares")
                    price_str = val("transactionPricePerShare")

                    shares = float(re.sub(r"[^\d.]", "", shares_str)) if shares_str else 0.0
                    price = float(re.sub(r"[^\d.]", "", price_str)) if price_str else 0.0
                    total_value = shares * price

                    if total_value > 0 and code in ("P", "S"):
                        transactions.append({
                            "ticker": ticker,
                            "company": company,
                            "insider_name": insider_name,
                            "insider_title": insider_title,
                            "transaction_date": date_str,
                            "transaction_type": "BUY" if code == "P" else "SELL",
                            "transaction_code": code,
                            "shares": shares,
                            "price_per_share": price,
                            "total_value": total_value,
                            "is_10b5_plan": is_10b5_plan,
                            "source_url": source_url,
                        })
                except Exception as e:
                    logger.debug(f"Error parsing transaction node: {e}")

        except Exception as e:
            logger.debug(f"Error parsing Form 4 XML: {e}")

        return transactions

    def get_top_insider_trades(self, hours: int = 24, top_n: int = 5) -> list[dict]:
        """
        Fetch all Form 4 filings in the last N hours, persist to SQLite,
        run insider-trading-analyzer filters, return top_n by dollar value.
        """
        logger.info("Starting SEC insider trading data collection...")
        filings = self.fetch_recent_form4_filings(hours=hours)

        all_transactions = []
        for i, filing in enumerate(filings[:50]):
            href = filing.get("filing_href", "")
            if not href:
                continue
            logger.debug(f"Parsing filing {i+1}/{min(len(filings), 50)}: {href}")
            txns = self.parse_form4_transaction(href)
            for t in txns:
                t["filing_updated"] = filing.get("updated", "")
            all_transactions.extend(txns)

        if not all_transactions:
            logger.warning("No transactions found — using cached data")
            return self._load_cached_data()

        # Persist to SQLite
        insert_trades(self.db, all_transactions)

        # Apply big-money filter (>$50k purchases, inspired by insider-trading-analyzer)
        significant = find_big_money_trades(self.db, min_value=50_000, transaction_type="BUY", days=hours // 24 + 1)
        significant += find_big_money_trades(self.db, min_value=50_000, transaction_type="SELL", days=hours // 24 + 1)

        # Fall back to raw sort if filter yields nothing
        if not significant:
            significant = sorted(all_transactions, key=lambda x: x["total_value"], reverse=True)

        # Deduplicate
        seen = set()
        unique = []
        for t in sorted(significant, key=lambda x: x.get("total_value", 0), reverse=True):
            key = (t.get("ticker"), t.get("insider_name"), t.get("transaction_date"))
            if key not in seen and t.get("ticker"):
                seen.add(key)
                unique.append(t)

        top = unique[:top_n]

        # Attach signal classification (TradeSignal-inspired)
        ticker_trades = defaultdict(list)
        for t in all_transactions:
            ticker_trades[t.get("ticker", "")].append(t)
        for t in top:
            t["signal"] = classify_signal(ticker_trades[t["ticker"]])

        # Cluster buy info
        clusters = find_cluster_buys(self.db, min_insiders=2, window_days=7)
        cluster_tickers = {c["ticker"] for c in clusters}
        for t in top:
            t["cluster_buy"] = t["ticker"] in cluster_tickers

        self._save_data(top)
        logger.info(f"Top {len(top)} insider trades: {[t['ticker'] for t in top]}")
        return top

    def get_signal_report(self) -> list[dict]:
        """Run all insider-trading-analyzer filters and return a combined report."""
        return {
            "cluster_buys": find_cluster_buys(self.db),
            "big_money_buys": find_big_money_trades(self.db, min_value=500_000, transaction_type="BUY"),
            "big_money_sells": find_big_money_trades(self.db, min_value=500_000, transaction_type="SELL"),
            "repeated_buyers": find_repeated_buyers(self.db),
        }

    def _save_data(self, data: list[dict]):
        path = Path(SEC_DATA_DIR) / "latest_insider_trades.json"
        with open(path, "w") as f:
            json.dump({"fetched_at": datetime.utcnow().isoformat(), "trades": data}, f, indent=2, default=str)
        logger.info(f"Saved SEC data to {path}")

    def _load_cached_data(self) -> list[dict]:
        path = Path(SEC_DATA_DIR) / "latest_insider_trades.json"
        if path.exists():
            with open(path) as f:
                return json.load(f).get("trades", [])
        return []
