"""
Hermes-style function calling agent powered by OpenRouter.

Implements the NousResearch Hermes function calling format:
  - Tools defined in <tools> XML in the system prompt
  - Model outputs <tool_call>{"name": ..., "arguments": ...}</tool_call>
  - Results fed back as tool_response messages
  - ReAct loop: Reason → Act → Observe → (repeat) → Final answer

Closed learning loop: the agent observes its own tool outputs and iterates
until it has enough information to answer confidently.
"""
import json
import logging
import re
from typing import Any, Callable, Optional

from openai import OpenAI

from config import MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

HERMES_SYSTEM_PROMPT = """You are a financial intelligence assistant for CrowdWisdomTrading.
You have access to real-time SEC insider trading data and social media sentiment from X/Twitter.

You MUST answer questions ONLY based on the provided context. If the answer is not in the context, say so clearly.

When the user asks for a chart or visualization, use the appropriate chart tool.
When answering about specific tickers or trades, cite the data source.

{tools_section}

{few_shot_section}

IMPORTANT RULES:
1. Only cite information present in the SEC filings and tweet data provided.
2. Never speculate or use general financial knowledge not grounded in the data.
3. If asked for a chart, always call the relevant chart tool.
4. Be concise and factual. Format numbers clearly (e.g., $1.2M, not 1200000).
"""

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


class HermesAgent:
    """
    Hermes-format function calling agent with ReAct loop.

    The closed learning loop is implemented at two levels:
    1. Within a single session: the agent iterates (Observe → Reason → Act)
       until it accumulates enough tool results to answer confidently.
    2. Across sessions: LearningLoop stores feedback and re-ranks retrieval.
    """

    def __init__(self, tools: dict[str, Callable], max_iterations: int = 5):
        """
        Args:
            tools: Dict mapping tool name → Python function
            max_iterations: Max number of tool call cycles per query
        """
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_schemas = self._build_tool_schemas()

    def _build_tool_schemas(self) -> list[dict]:
        """Build OpenAI-compatible tool schemas from registered tools."""
        schemas = []
        for name, fn in self.tools.items():
            schema = getattr(fn, "_schema", None)
            if schema:
                schemas.append(schema)
        return schemas

    def _tools_xml(self) -> str:
        """Format tools as XML for the Hermes system prompt."""
        if not self.tool_schemas:
            return ""
        tools_json = json.dumps(self.tool_schemas, indent=2)
        return f"You have access to the following tools:\n<tools>\n{tools_json}\n</tools>\n"

    def chat(
        self,
        user_message: str,
        context: str,
        conversation_history: list[dict],
        few_shot_examples: str = "",
        session_id: str = "",
    ) -> tuple[str, list[str]]:
        """
        Run the Hermes ReAct loop for a user message.

        Args:
            user_message: The user's query
            context: Retrieved RAG context
            conversation_history: Prior turns in this session
            few_shot_examples: Optional learned examples from LearningLoop
            session_id: For logging

        Returns:
            (final_answer, list_of_chart_base64_images)
        """
        system = HERMES_SYSTEM_PROMPT.format(
            tools_section=self._tools_xml(),
            few_shot_section=few_shot_examples,
        )

        # Build the augmented user message with RAG context
        augmented_message = (
            f"<context>\n{context}\n</context>\n\n"
            f"User question: {user_message}"
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history[-6:])  # Keep last 3 turns
        messages.append({"role": "user", "content": augmented_message})

        charts = []
        iterations = 0

        # ReAct loop
        while iterations < self.max_iterations:
            iterations += 1
            logger.debug(f"ReAct iteration {iterations}/{self.max_iterations}")

            response = self._call_llm(messages)
            content = response.choices[0].message.content or ""

            # Check for tool calls in Hermes format
            tool_calls = TOOL_CALL_PATTERN.findall(content)

            if not tool_calls:
                # No tool calls — this is the final answer
                # Strip any residual XML tags
                final = re.sub(r"<[^>]+>", "", content).strip()
                return final, charts

            # Execute tool calls
            messages.append({"role": "assistant", "content": content})
            tool_results = []

            for tc_json in tool_calls:
                try:
                    tc = json.loads(tc_json)
                    tool_name = tc.get("name", "")
                    arguments = tc.get("arguments", {})
                    logger.info(f"Calling tool: {tool_name} with {list(arguments.keys())}")

                    if tool_name in self.tools:
                        result = self.tools[tool_name](**arguments)
                        # If result is a base64 image, collect it
                        if isinstance(result, str) and len(result) > 500 and _is_base64(result):
                            charts.append(result)
                            result = "[Chart generated and displayed to user]"
                        tool_results.append(
                            f"<tool_response>\n"
                            f"<tool_name>{tool_name}</tool_name>\n"
                            f"<result>{json.dumps(result)}</result>\n"
                            f"</tool_response>"
                        )
                    else:
                        tool_results.append(
                            f"<tool_response><tool_name>{tool_name}</tool_name>"
                            f"<result>Tool not found</result></tool_response>"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid tool call JSON: {e}")
                    tool_results.append(
                        "<tool_response><result>Invalid tool call format</result></tool_response>"
                    )

            # Feed tool results back to model
            messages.append({
                "role": "user",
                "content": "\n".join(tool_results),
            })

        # If we hit max iterations, return whatever the last response was
        logger.warning("Reached max ReAct iterations")
        return "I processed the available data but ran out of reasoning steps. Please try a more specific question.", charts

    def _call_llm(self, messages: list[dict]):
        """Call the OpenRouter LLM."""
        try:
            return self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                extra_headers={
                    "HTTP-Referer": "https://crowdwisdomtrading.com",
                    "X-Title": "CrowdWisdomTrading AI",
                },
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


def tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as a Hermes tool with schema."""
    def decorator(fn: Callable) -> Callable:
        fn._schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [k for k, v in parameters.items() if v.get("required", False)],
                },
            },
        }
        fn._tool_name = name
        return fn
    return decorator


def _is_base64(s: str) -> bool:
    """Heuristic check if a string looks like a base64-encoded image."""
    import base64
    try:
        base64.b64decode(s[:100])
        return True
    except Exception:
        return False
