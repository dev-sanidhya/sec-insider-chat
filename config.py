import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.getenv("MODEL", "nousresearch/hermes-3-llama-3.1-8b:free")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "CrowdWisdomTrading contact@example.com")
PORT = int(os.getenv("PORT", "7860"))

# Optional premium data sources (from referenced repos)
SEC_API_KEY = os.getenv("SEC_API_KEY", "")          # sec-api.io (janlukasschroeder/sec-api-python)
ENABLE_SEC_AGENTKIT = os.getenv("ENABLE_SEC_AGENTKIT", "false").lower() == "true"  # sec-edgar-agentkit

# SEC EDGAR endpoints
SEC_EDGAR_BASE = "https://data.sec.gov"
SEC_EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
SEC_FORM4_RSS = "https://www.sec.gov/cgi-bin/browse-edgar"

# APIFY actor for Twitter/X scraping (free with platform credits)
APIFY_TWITTER_ACTOR = "web.harvester/easy-twitter-search-scraper"

# RAG settings
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast, free local model
CHUNK_SIZE_TWEETS = 1          # 1 tweet per chunk
CHUNK_SIZE_SEC_FILING = 1      # 1 transaction per chunk
TOP_K_RESULTS = 8              # RAG retrieval top-k

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEC_DATA_DIR = os.path.join(DATA_DIR, "sec_data")
TWEETS_DIR = os.path.join(DATA_DIR, "tweets")
CHARTS_DIR = os.path.join(DATA_DIR, "charts")
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback")

# Ensure dirs exist
for _d in [SEC_DATA_DIR, TWEETS_DIR, CHARTS_DIR, FEEDBACK_DIR]:
    os.makedirs(_d, exist_ok=True)
