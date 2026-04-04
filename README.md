# CrowdWisdomTrading AI SEC Chat

A backend Python chatbot that combines **SEC Form 4 insider trading data** with **X/Twitter sentiment analysis**, powered by a **Hermes-style function calling agent** and **RAG** (ChromaDB).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Gradio Chat UI                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│           Hermes Agent (OpenRouter LLM)                 │
│    ReAct Loop: Observe → Reason → Act → Feedback        │
└──────┬───────────────┬───────────────┬──────────────────┘
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  RAG Search │ │ Chart Tools │ │ Data Tools  │
│  (ChromaDB) │ │ (matplotlib)│ │ (SEC/APIFY) │
└──────┬──────┘ └─────────────┘ └──────┬──────┘
       │                               │
┌──────▼──────────────────────────────▼──────────────────┐
│           ChromaDB (Persistent Vector Store)           │
│    SEC Collection  │  Tweets Collection                │
└─────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Closed Learning Loop                       │
│  User Feedback → Memory → Re-ranking → Better Answers  │
└─────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- **OPENROUTER_API_KEY**: Get at [openrouter.ai](https://openrouter.ai) (free tier available)
- **APIFY_API_TOKEN**: Get at [console.apify.com](https://console.apify.com) (free tier: $5 credit/month)
- **SEC_USER_AGENT**: Your name + email (required by SEC EDGAR)

### 3. Run the data pipeline
```bash
python -m flow.pipeline
```
This will:
1. Fetch top 5 SEC Form 4 insider trades (last 24h)
2. Scrape X/Twitter for each ticker (last 7 days via APIFY)
3. Index everything into ChromaDB

### 4. Launch the chatbot
```bash
python main.py
```
Open `http://localhost:7860` in your browser.

## Project Structure
```
ai-sec-chat/
├── main.py                  # Gradio chatbot UI + tool definitions
├── config.py                # Configuration (env vars, paths)
├── requirements.txt
├── agents/
│   └── hermes_agent.py      # Hermes-style function calling agent
├── tools/
│   ├── sec_tools.py         # SEC EDGAR Form 4 fetcher + parser
│   ├── apify_tools.py       # APIFY Twitter scraper
│   └── chart_tools.py       # matplotlib chart generation
├── rag/
│   ├── indexer.py           # ChromaDB indexer (chunking strategy)
│   └── retriever.py         # Semantic retrieval + context formatting
├── flow/
│   └── pipeline.py          # Data collection orchestration
├── feedback/
│   └── learning_loop.py     # Closed learning loop (feedback → memory)
├── data/
│   ├── sec_data/            # Cached SEC filings
│   ├── tweets/              # Cached tweet data
│   └── charts/              # Generated chart PNGs
└── examples/
    ├── generate_sample_output.py
    ├── sample_sec_trades.json
    ├── sample_tweets.json
    └── sample_qa_output.json
```

## RAG Chunking Strategy

| Data Type | Chunk Size | Chunk Content | Metadata |
|-----------|-----------|---------------|----------|
| SEC Form 4 | 1 transaction per chunk | Human-readable sentence describing the trade | ticker, insider, date, value, type, URL |
| Tweets | 1 tweet per chunk | Tweet text with author and engagement stats | ticker, author, date, likes, retweets, engagement |

**Why this chunking?**
- SEC transactions are atomic facts — combining them loses precision
- Tweets are self-contained opinions — per-tweet chunking enables fine-grained sentiment retrieval
- Human-readable summaries are embedded (not raw JSON) for better semantic search

## Hermes Agent & Closed Learning Loop

The agent uses the [NousResearch Hermes](https://github.com/nousresearch/hermes-agent) function calling format:

```
System: You have tools: <tools>[...]</tools>
User: <context>RAG results</context> Question
Assistant: <tool_call>{"name": "search_tweets", "arguments": {"ticker": "NVDA"}}</tool_call>
Tool: <tool_response>...</tool_response>
Assistant: Based on the data, NVDA has...
```

**Closed Learning Loop (Observe → Reason → Act → Feedback → Update):**
1. Agent executes tool calls iteratively until it has enough context
2. User rates the answer (👍 / 👎)
3. Positive feedback: answer stored as few-shot example, ticker weights boosted
4. Negative feedback: query pattern logged, ticker weights reduced
5. Future queries: retrieval re-ranked using learned weights + few-shot examples injected into system prompt

## Available Chart Types

- **Insider Trades Bar Chart**: Top 5 trades by dollar value (color-coded BUY/SELL)
- **Tweet Volume Over Time**: Daily tweet count per ticker
- **Sentiment Donut Chart**: Positive/Negative/Neutral breakdown (keyword-based)
- **Trade Timeline**: Scatter plot of trades over time
- **Engagement Comparison**: Likes + retweets per ticker

## Sample Questions

- "What are the top insider trades in the last 24 hours?"
- "Show me a chart of insider trading activity"
- "What is the social media sentiment for NVDA?"
- "Show tweet volume chart for TSLA"
- "Which insiders made the biggest buys today?"
- "Compare social engagement across all tickers"

## Data Sources

- **SEC EDGAR**: Official SEC EDGAR API (`data.sec.gov`) — Form 4 filings (free, no API key needed)
- **APIFY**: [apidojo/tweet-scraper](https://apify.com/apidojo/tweet-scraper) — X/Twitter scraping
- **OpenRouter**: LLM inference ([nousresearch/hermes-3-llama-3.1-8b:free](https://openrouter.ai))

## Scaling Considerations

- **Data freshness**: Run pipeline as a cron job (`0 * * * * python -m flow.pipeline`) for hourly updates
- **More tickers**: Increase `top_n_trades` in `pipeline.run()` (currently 5)
- **More tweets**: Increase `max_tweets_per_ticker` (currently 100)
- **Production DB**: Replace ChromaDB with Pinecone or Weaviate for scale
- **Async**: Wrap APIFY calls in `asyncio` for parallel ticker scraping

## Logging

All components use Python's standard `logging` module with `rich` formatting.
Set `LOG_LEVEL=DEBUG` for verbose output.
