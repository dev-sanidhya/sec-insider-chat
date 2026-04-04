"""
Closed learning loop: collects user feedback, logs query/answer pairs,
and uses feedback to improve future retrieval (re-ranking + memory).

Hermes-inspired loop: Observe → Reason → Act → Feedback → Update
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import FEEDBACK_DIR

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path(FEEDBACK_DIR) / "feedback_store.json"
MEMORY_FILE = Path(FEEDBACK_DIR) / "agent_memory.json"


class LearningLoop:
    """
    Implements a closed learning loop for the agent.

    Functionality:
    1. Log every Q&A interaction with metadata
    2. Collect user thumbs-up/thumbs-down ratings
    3. Identify patterns in failed queries (low-rated answers)
    4. Build a memory of successful query-answer pairs for few-shot examples
    5. Adjust retrieval strategy based on feedback (ticker-level weighting)
    """

    def __init__(self):
        self.feedback_store = self._load_json(FEEDBACK_FILE, default=[])
        self.memory = self._load_json(MEMORY_FILE, default={
            "successful_examples": [],
            "ticker_weights": {},
            "query_patterns": {},
        })

    # ------------------------------------------------------------------ #
    # Logging interactions
    # ------------------------------------------------------------------ #

    def log_interaction(
        self,
        query: str,
        answer: str,
        context_used: list[dict],
        session_id: str = "",
    ) -> str:
        """Log a Q&A interaction. Returns a unique interaction ID."""
        interaction_id = f"int_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        entry = {
            "id": interaction_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "answer": answer,
            "context_chunks": len(context_used),
            "tickers_referenced": list({
                r.get("metadata", {}).get("ticker", "") for r in context_used if r.get("metadata", {}).get("ticker")
            }),
            "rating": None,  # Set later via record_feedback
            "feedback_text": None,
        }
        self.feedback_store.append(entry)
        self._save_json(FEEDBACK_FILE, self.feedback_store)
        return interaction_id

    def record_feedback(
        self,
        interaction_id: str,
        rating: int,  # 1 = thumbs up, -1 = thumbs down, 0 = neutral
        feedback_text: Optional[str] = None,
    ):
        """Record user feedback for a past interaction."""
        for entry in self.feedback_store:
            if entry["id"] == interaction_id:
                entry["rating"] = rating
                entry["feedback_text"] = feedback_text
                break
        self._save_json(FEEDBACK_FILE, self.feedback_store)
        self._update_memory(interaction_id, rating)
        logger.info(f"Feedback recorded for {interaction_id}: rating={rating}")

    # ------------------------------------------------------------------ #
    # Memory update (closed loop)
    # ------------------------------------------------------------------ #

    def _update_memory(self, interaction_id: str, rating: int):
        """
        Update agent memory based on feedback.

        Positive feedback (rating=1):
          - Add to successful_examples for future few-shot prompting
          - Boost ticker weights for the referenced tickers

        Negative feedback (rating=-1):
          - Log failed patterns for analysis
          - Reduce confidence on involved tickers
        """
        entry = next((e for e in self.feedback_store if e["id"] == interaction_id), None)
        if not entry:
            return

        tickers = entry.get("tickers_referenced", [])

        if rating == 1:
            # Add to successful examples (cap at 10 most recent)
            example = {"query": entry["query"], "answer": entry["answer"][:500]}
            examples = self.memory["successful_examples"]
            examples.append(example)
            self.memory["successful_examples"] = examples[-10:]

            # Boost ticker weights
            for ticker in tickers:
                self.memory["ticker_weights"][ticker] = (
                    self.memory["ticker_weights"].get(ticker, 1.0) + 0.1
                )

        elif rating == -1:
            # Log failed query pattern
            pattern = entry["query"][:100]
            self.memory["query_patterns"][pattern] = {
                "count": self.memory["query_patterns"].get(pattern, {}).get("count", 0) + 1,
                "last_seen": datetime.utcnow().isoformat(),
            }

            # Reduce ticker weights slightly
            for ticker in tickers:
                weight = self.memory["ticker_weights"].get(ticker, 1.0)
                self.memory["ticker_weights"][ticker] = max(0.5, weight - 0.05)

        self._save_json(MEMORY_FILE, self.memory)

    # ------------------------------------------------------------------ #
    # Retrieval enhancement
    # ------------------------------------------------------------------ #

    def get_few_shot_examples(self, n: int = 2) -> str:
        """Return the best few-shot Q&A examples for prompt injection."""
        examples = self.memory.get("successful_examples", [])[-n:]
        if not examples:
            return ""
        lines = ["--- Successful past answers for reference ---"]
        for ex in examples:
            lines.append(f"Q: {ex['query']}\nA: {ex['answer']}\n")
        return "\n".join(lines)

    def rerank_results(self, results: list[dict]) -> list[dict]:
        """
        Re-rank RAG results using learned ticker weights.
        Results with positively-reinforced tickers float to the top.
        """
        weights = self.memory.get("ticker_weights", {})
        if not weights:
            return results

        for r in results:
            ticker = r.get("metadata", {}).get("ticker", "")
            w = weights.get(ticker, 1.0)
            r["relevance_score"] = r.get("relevance_score", 0.5) * w

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Stats / reporting
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Return learning loop statistics."""
        total = len(self.feedback_store)
        rated = [e for e in self.feedback_store if e.get("rating") is not None]
        positive = sum(1 for e in rated if e.get("rating", 0) > 0)
        negative = sum(1 for e in rated if e.get("rating", 0) < 0)
        return {
            "total_interactions": total,
            "rated": len(rated),
            "positive_ratings": positive,
            "negative_ratings": negative,
            "satisfaction_rate": f"{(positive / len(rated) * 100):.1f}%" if rated else "N/A",
            "successful_examples_in_memory": len(self.memory.get("successful_examples", [])),
            "ticker_weights": self.memory.get("ticker_weights", {}),
        }

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return default

    @staticmethod
    def _save_json(path: Path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
