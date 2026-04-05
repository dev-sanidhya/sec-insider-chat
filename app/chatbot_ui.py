"""
CrowdWisdomTrading RAG Chatbot — Flask (no Gradio dependency issues)
"""
import sys, io, json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template_string
from rag.indexer import RAGIndexer

app = Flask(__name__)

indexer   = RAGIndexer()
sec_col   = indexer.client.get_or_create_collection("sec_insider_trades")
tweet_col = indexer.client.get_or_create_collection("tweets")


def rag_answer(message: str) -> str:
    sec_r   = sec_col.query(query_texts=[message], n_results=3)
    tweet_r = tweet_col.query(query_texts=[message], n_results=3)

    sec_docs   = sec_r["documents"][0]   if sec_r["documents"]   else []
    tweet_docs = tweet_r["documents"][0] if tweet_r["documents"] else []
    sec_meta   = sec_r["metadatas"][0]   if sec_r["metadatas"]   else []
    tweet_meta = tweet_r["metadatas"][0] if tweet_r["metadatas"] else []

    lines = []

    if sec_docs:
        lines.append("📊 SEC INSIDER TRADES (EDGAR Form 4)")
        for i, (doc, m) in enumerate(zip(sec_docs, sec_meta), 1):
            val = m.get("total_value", 0)
            lines.append("\n{}. {} | {} | {} | ${:,.0f} | {}".format(
                i, m.get("ticker",""), m.get("insider_name",""),
                m.get("transaction_type",""), val, m.get("transaction_date","")
            ))
            lines.append("   " + doc[:220])

    if tweet_docs:
        lines.append("\n🐦 SOCIAL SENTIMENT (X / Twitter via APIFY)")
        for i, (doc, m) in enumerate(zip(tweet_docs, tweet_meta), 1):
            lines.append("\n{}. @{} on ${} | {} likes | {} retweets".format(
                i, m.get("author","unknown"), m.get("ticker",""),
                m.get("likes",0), m.get("retweets",0)
            ))
            lines.append("   " + doc[:220])

    if not lines:
        return "No data found. Try asking about CRWV, NTRA, or SYRE."

    return "\n".join(lines)


HTML = """<!DOCTYPE html>
<html>
<head>
  <title>CrowdWisdomTrading RAG Chatbot</title>
  <meta charset="utf-8">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f1117; color: #e0e0e0; height: 100vh; display: flex;
           flex-direction: column; }
    header { padding: 16px 24px; background: #1a1d27; border-bottom: 1px solid #2d3142; }
    header h1 { font-size: 1.3rem; color: #fff; }
    header p  { font-size: 0.82rem; color: #8b8fa8; margin-top: 4px; }
    .stats { display: flex; gap: 16px; margin-top: 8px; }
    .stat  { background: #252836; padding: 4px 12px; border-radius: 20px;
             font-size: 0.78rem; color: #a78bfa; }
    #chat  { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex;
             flex-direction: column; gap: 14px; }
    .msg   { max-width: 80%; padding: 12px 16px; border-radius: 12px;
             white-space: pre-wrap; font-size: 0.88rem; line-height: 1.6; }
    .user  { background: #4f46e5; color: #fff; align-self: flex-end;
             border-bottom-right-radius: 4px; }
    .bot   { background: #1e2130; color: #e0e0e0; align-self: flex-start;
             border-bottom-left-radius: 4px; border: 1px solid #2d3142; }
    .typing { color: #6b7280; font-style: italic; }
    footer  { padding: 14px 24px; background: #1a1d27;
              border-top: 1px solid #2d3142; display: flex; gap: 10px; }
    #input  { flex: 1; background: #252836; border: 1px solid #3d4158;
              color: #e0e0e0; padding: 10px 16px; border-radius: 8px;
              font-size: 0.9rem; outline: none; }
    #input:focus { border-color: #4f46e5; }
    #send   { background: #4f46e5; color: #fff; border: none; padding: 10px 20px;
              border-radius: 8px; cursor: pointer; font-size: 0.9rem; }
    #send:hover { background: #4338ca; }
    .examples { padding: 0 24px 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .ex-btn { background: #1e2130; border: 1px solid #3d4158; color: #a78bfa;
              padding: 5px 12px; border-radius: 20px; cursor: pointer;
              font-size: 0.78rem; }
    .ex-btn:hover { background: #252836; }
  </style>
</head>
<body>
  <header>
    <h1>CrowdWisdomTrading — Insider Intelligence Chatbot</h1>
    <p>Answers grounded in real SEC Form 4 filings + live X/Twitter sentiment</p>
    <div class="stats">
      <span class="stat">SEC Trades: {{ sec_count }}</span>
      <span class="stat">Tweet chunks: {{ tweet_count }}</span>
      <span class="stat">Total indexed: {{ total }}</span>
    </div>
  </header>

  <div id="chat">
    <div class="msg bot">Hi! Ask me about insider trades or social sentiment for CRWV, NTRA, or SYRE.</div>
  </div>

  <div class="examples">
    <button class="ex-btn" onclick="ask('What insider trades happened in CRWV?')">CRWV insider trades</button>
    <button class="ex-btn" onclick="ask('What do people say about NTRA on social media?')">NTRA social sentiment</button>
    <button class="ex-btn" onclick="ask('Which insider sold the most shares recently?')">Biggest insider sell</button>
    <button class="ex-btn" onclick="ask('Show me SYRE recent activity')">SYRE activity</button>
  </div>

  <footer>
    <input id="input" type="text" placeholder="Ask about insider trades or social sentiment..."
           onkeydown="if(event.key==='Enter') sendMsg()">
    <button id="send" onclick="sendMsg()">Send</button>
  </footer>

<script>
  const chat = document.getElementById('chat');
  const input = document.getElementById('input');

  function addMsg(text, cls) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    d.textContent = text;
    chat.appendChild(d);
    chat.scrollTop = chat.scrollHeight;
    return d;
  }

  function ask(q) { input.value = q; sendMsg(); }

  async function sendMsg() {
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    addMsg(msg, 'user');
    const typing = addMsg('Thinking...', 'bot typing');
    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg})
      });
      const data = await resp.json();
      typing.className = 'msg bot';
      typing.textContent = data.answer;
    } catch(e) {
      typing.textContent = 'Error: ' + e.message;
    }
    chat.scrollTop = chat.scrollHeight;
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    stats = indexer.get_stats()
    return render_template_string(
        HTML,
        sec_count=stats.get("sec_insider_trades", 0),
        tweet_count=stats.get("tweets", 0),
        total=sum(stats.values()),
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"answer": "Please ask a question."})
    answer = rag_answer(message)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    stats = indexer.get_stats()
    print("\nSEC chunks:   ", stats.get("sec_insider_trades", 0))
    print("Tweet chunks: ", stats.get("tweets", 0))
    print("Open:          http://localhost:7860\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
