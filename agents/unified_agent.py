import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def _coerce_confidence(value: Any) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _normalize_attribute_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    if "name" in item:
        name = str(item.get("name") or "").strip()
        if not name:
            return None
        return {
            "name": name,
            "value": item.get("value"),
            "confidence": _coerce_confidence(item.get("confidence", 0)),
            "source_excerpt": item.get("source_excerpt"),
        }

    # Fallback: {"Supply Voltage": "24"}
    if len(item) == 1:
        (name, value), = item.items()
        name = str(name).strip()
        if not name:
            return None
        return {
            "name": name,
            "value": value,
            "confidence": _coerce_confidence(item.get("confidence", 0)),
            "source_excerpt": None,
        }

    return None


def _normalize_attributes(raw_attributes: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    if isinstance(raw_attributes, list):
        for item in raw_attributes:
            entry = _normalize_attribute_item(item)
            if entry is not None:
                normalized.append(entry)
        return normalized

    if isinstance(raw_attributes, dict):
        if "name" in raw_attributes and "value" in raw_attributes:
            entry = _normalize_attribute_item(raw_attributes)
            if entry is not None:
                normalized.append(entry)
            return normalized

        for key, value in raw_attributes.items():
            if key == "attributes":
                normalized.extend(_normalize_attributes(value))
                continue
            if isinstance(value, dict) and "value" in value:
                entry = {
                    "name": str(key).strip(),
                    "value": value.get("value"),
                    "confidence": _coerce_confidence(value.get("confidence", 0)),
                    "source_excerpt": value.get("source_excerpt"),
                }
            else:
                entry = {
                    "name": str(key).strip(),
                    "value": value,
                    "confidence": 0,
                    "source_excerpt": None,
                }
            if entry["name"]:
                normalized.append(entry)

    return normalized


class UnifiedAgent:
    """Single unified agent that extracts dynamic attributes in one LLM call."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
            max_retries=0,  # disable built-in retry — orchestrator handles backoff
        )

    def extract(self, chunks: list[str]) -> dict:
        if not chunks:
            return {"attributes": []}

        prompt = self.get_prompt(chunks)
        response = self.llm.invoke(prompt)
        return self._parse_single_response(response.content)

    def extract_batch(self, product_contexts: dict[str, str]) -> dict[str, dict]:
        if not product_contexts:
            return {}
        prompt = self.get_batch_prompt(product_contexts)
        response = self.llm.invoke(prompt)
        return self._parse_batch_response(response.content, list(product_contexts.keys()))

    def _parse_single_response(self, content: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return {"attributes": []}

        raw_attrs = parsed.get("attributes", parsed)
        return {"attributes": _normalize_attributes(raw_attrs)}

    def _parse_batch_response(self, content: str, product_ids: list[str]) -> dict[str, dict]:
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return {pid: {"attributes": []} for pid in product_ids}

        products = parsed.get("products", {}) if isinstance(parsed, dict) else {}
        output: dict[str, dict] = {}
        for pid in product_ids:
            product_payload = products.get(pid, {}) if isinstance(products, dict) else {}
            if isinstance(product_payload, dict):
                raw_attrs = product_payload.get("attributes", product_payload)
            else:
                raw_attrs = []
            output[pid] = {"attributes": _normalize_attributes(raw_attrs)}
        return output

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You extract technical product attributes from text.

Return ONLY valid JSON in this shape:
{{
  "attributes": [
    {{"name": "Supply Voltage", "value": "24", "confidence": 90, "source_excerpt": "Supply voltage: 24V"}}
  ]
}}

Rules:
- Use dynamic attribute names exactly as seen in text (Title Case when appropriate).
- Include only attributes supported by evidence in text.
- If an attribute has multiple values/variants, emit multiple entries with the same name.
- Do not output a fixed template and do not include unknown placeholders.
- No markdown.

Text:
{chunks_text}"""

    def get_batch_prompt(self, product_contexts: dict[str, str]) -> str:
        blocks = []
        for product_id, context in product_contexts.items():
            blocks.append(f"PRODUCT_ID: {product_id}\nCONTEXT:\n{context}")
        payload = "\n\n".join(blocks)

        return f"""You extract technical product attributes from compact evidence.

Process every product block and return ONLY valid JSON in this exact shape:
{{
  "products": {{
    "<product_id>": {{
      "attributes": [
        {{"name": "Supply Voltage", "value": "24", "confidence": 90, "source_excerpt": "Supply voltage: 24V"}}
      ]
    }}
  }}
}}

Rules:
- Include all provided product IDs under products.
- Attribute names must be dynamic (not a fixed key set).
- Include only evidence-backed attributes.
- Keep duplicate names when distinct values exist (for example multiple control signals).
- No markdown.

PRODUCT BLOCKS:
{payload}"""
