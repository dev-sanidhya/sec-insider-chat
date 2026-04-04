"""
SEC EDGAR Form 4 insider trading data fetcher.
Pulls filings from the last 24 hours and parses transaction values.
"""
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import SEC_EDGAR_BASE, SEC_FORM4_RSS, SEC_USER_AGENT, SEC_DATA_DIR

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


class SECInsiderTradingFetcher:
    """Fetches and parses SEC Form 4 insider trading filings."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get(self, url: str, params: dict = None) -> requests.Response:
        time.sleep(0.1)  # SEC rate limit: 10 req/sec
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp

    def fetch_recent_form4_filings(self, hours: int = 24) -> list[dict]:
        """
        Fetch Form 4 filings from the last N hours via SEC EDGAR full-text search.
        Returns a list of filing metadata dicts.
        """
        logger.info(f"Fetching Form 4 filings from the last {hours} hours...")
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
        try:
            resp = self._get(SEC_FORM4_RSS, params=params)
            soup = BeautifulSoup(resp.content, "xml")
            entries = soup.find_all("entry")
            logger.info(f"Found {len(entries)} Form 4 entries in RSS feed")

            for entry in entries:
                try:
                    filing = self._parse_rss_entry(entry)
                    if filing:
                        filings.append(filing)
                except Exception as e:
                    logger.debug(f"Error parsing entry: {e}")
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {e}")

        return filings

    def _parse_rss_entry(self, entry) -> Optional[dict]:
        """Parse a single RSS entry into a filing metadata dict."""
        title = entry.find("title")
        updated = entry.find("updated")
        filing_href = entry.find("filing-href")
        company_name = entry.find("company-name")
        cik = entry.find("cik")

        if not all([title, filing_href]):
            return None

        return {
            "title": title.text.strip() if title else "",
            "updated": updated.text.strip() if updated else "",
            "filing_href": filing_href.text.strip() if filing_href else "",
            "company_name": company_name.text.strip() if company_name else "",
            "cik": cik.text.strip() if cik else "",
        }

    def parse_form4_transaction(self, filing_href: str) -> list[dict]:
        """
        Parse a Form 4 filing index page, then download and parse the XML.
        Returns list of transaction dicts with value in dollars.
        """
        transactions = []
        try:
            # Get the filing index page
            resp = self._get(filing_href)
            soup = BeautifulSoup(resp.content, "html.parser")

            # Find the XML document link
            xml_link = None
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.endswith(".xml") and "xbrl" not in href.lower():
                    xml_link = f"https://www.sec.gov{href}" if href.startswith("/") else href
                    break

            if not xml_link:
                return transactions

            # Parse the XML Form 4
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

            # Issuer / company info
            issuer = soup.find("issuer")
            ticker = ""
            company = ""
            if issuer:
                ticker_tag = issuer.find("issuerTradingSymbol")
                name_tag = issuer.find("issuerName")
                ticker = ticker_tag.text.strip() if ticker_tag else ""
                company = name_tag.text.strip() if name_tag else ""

            # Reporting owner (insider)
            owner = soup.find("reportingOwner")
            insider_name = ""
            insider_title = ""
            if owner:
                name_tag = owner.find("rptOwnerName")
                title_tag = owner.find("officerTitle")
                insider_name = name_tag.text.strip() if name_tag else ""
                insider_title = title_tag.text.strip() if title_tag else ""

            # Non-derivative transactions (actual stock buys/sells)
            for txn in soup.find_all("nonDerivativeTransaction"):
                try:
                    date_tag = txn.find("transactionDate")
                    code_tag = txn.find("transactionCode")
                    shares_tag = txn.find("transactionShares")
                    price_tag = txn.find("transactionPricePerShare")
                    acquired_tag = txn.find("transactionAcquiredDisposedCode")

                    date_str = date_tag.find("value").text.strip() if date_tag else ""
                    code = code_tag.find("value").text.strip() if code_tag else ""
                    shares_val = shares_tag.find("value").text.strip() if shares_tag else "0"
                    price_val = price_tag.find("value").text.strip() if price_tag else "0"
                    direction = acquired_tag.find("value").text.strip() if acquired_tag else ""

                    shares = float(re.sub(r"[^\d.]", "", shares_val)) if shares_val else 0.0
                    price = float(re.sub(r"[^\d.]", "", price_val)) if price_val else 0.0
                    total_value = shares * price

                    if total_value > 0 and code in ("P", "S"):  # Purchase or Sale
                        transactions.append({
                            "ticker": ticker,
                            "company": company,
                            "insider_name": insider_name,
                            "insider_title": insider_title,
                            "transaction_date": date_str,
                            "transaction_type": "BUY" if code == "P" else "SELL",
                            "shares": shares,
                            "price_per_share": price,
                            "total_value": total_value,
                            "direction": direction,
                            "source_url": source_url,
                        })
                except Exception as e:
                    logger.debug(f"Error parsing transaction: {e}")

        except Exception as e:
            logger.debug(f"Error parsing Form 4 XML: {e}")

        return transactions

    def get_top_insider_trades(self, hours: int = 24, top_n: int = 5) -> list[dict]:
        """
        Main entry point: fetch all Form 4 filings in the last N hours,
        parse them, return the top_n by total dollar value.
        """
        logger.info("Starting SEC insider trading data collection...")
        filings = self.fetch_recent_form4_filings(hours=hours)

        all_transactions = []
        for i, filing in enumerate(filings[:50]):  # Cap at 50 filings to avoid overload
            href = filing.get("filing_href", "")
            if not href:
                continue
            logger.debug(f"Parsing filing {i+1}/{min(len(filings), 50)}: {href}")
            txns = self.parse_form4_transaction(href)
            for t in txns:
                t["filing_updated"] = filing.get("updated", "")
            all_transactions.extend(txns)

        if not all_transactions:
            logger.warning("No transactions found. Using cached data if available.")
            return self._load_cached_data()

        # Sort by total_value descending, deduplicate by ticker+insider+date
        seen = set()
        unique_txns = []
        for t in sorted(all_transactions, key=lambda x: x["total_value"], reverse=True):
            key = (t["ticker"], t["insider_name"], t["transaction_date"])
            if key not in seen and t["ticker"]:
                seen.add(key)
                unique_txns.append(t)

        top = unique_txns[:top_n]
        self._save_data(top)
        logger.info(f"Top {len(top)} insider trades found: {[t['ticker'] for t in top]}")
        return top

    def _save_data(self, data: list[dict]):
        path = Path(SEC_DATA_DIR) / "latest_insider_trades.json"
        with open(path, "w") as f:
            json.dump({"fetched_at": datetime.utcnow().isoformat(), "trades": data}, f, indent=2)
        logger.info(f"Saved SEC data to {path}")

    def _load_cached_data(self) -> list[dict]:
        path = Path(SEC_DATA_DIR) / "latest_insider_trades.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Loaded cached SEC data from {path}")
            return data.get("trades", [])
        return []
