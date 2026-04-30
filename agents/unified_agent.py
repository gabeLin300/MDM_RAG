import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

ATTRIBUTES = [
    # Certifications
    "certifications", "standards_compliance", "regulatory_approvals",
    "safety_certifications", "environmental_certifications", "industry_certifications",
    # Connectivity
    "communication_protocols", "wired_interfaces", "ports",
    "network_capabilities", "data_rate", "bus_type",
    # Electrical
    "voltage_rating", "current_rating", "power_consumption", "power_supply", "frequency",
    # Environmental
    "operating_temperature", "storage_temperature", "humidity_range",
    "ingress_protection", "shock_resistance", "vibration_resistance", "altitude_rating",
    # Physical
    "dimensions", "weight", "material", "housing", "finish", "mounting_type", "enclosure_type",
]

_NULL_ENTRY = {"value": None, "confidence": 0, "source_excerpt": None}


class UnifiedAgent:
    """Extracts all product attributes with confidence scores in a single LLM call."""

    def __init__(self, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.model_name = model_name
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
        )

    def extract(self, chunks: list[str]) -> dict:
        """Extract all attributes from chunks in a single LLM call."""
        if not chunks:
            return {"attributes": {name: _NULL_ENTRY.copy() for name in ATTRIBUTES}}

        prompt = self._build_prompt(chunks)
        response = self.llm.invoke(prompt)
        return self._parse_response(response.content)

    def _build_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        example_found = '{"value": "CE, UL", "confidence": 92, "source_excerpt": "CE and UL certified for industrial use"}'
        example_null = '{"value": null, "confidence": 0, "source_excerpt": null}'
        return f"""You are a specialist extracting product specifications from Honeywell technical documents.

For each attribute return an object with three fields:
- "value": extracted string or null if not found
- "confidence": integer 0-100 (90-100 verbatim match, 70-89 clearly stated, 50-69 inferred, 1-49 uncertain, 0 not found)
- "source_excerpt": a direct quote of 20 words or fewer from the text that supports the value, or null

Return ONLY a raw JSON object — no markdown, no code blocks, no explanation.

Attributes to extract:
CERTIFICATIONS:  certifications, standards_compliance, regulatory_approvals, safety_certifications, environmental_certifications, industry_certifications
CONNECTIVITY:    communication_protocols, wired_interfaces, ports, network_capabilities, data_rate, bus_type
ELECTRICAL:      voltage_rating, current_rating, power_consumption, power_supply, frequency
ENVIRONMENTAL:   operating_temperature, storage_temperature, humidity_range, ingress_protection, shock_resistance, vibration_resistance, altitude_rating
PHYSICAL:        dimensions, weight, material, housing, finish, mounting_type, enclosure_type

Example output:
{{
  "attributes": {{
    "certifications": {example_found},
    "voltage_rating": {example_null}
  }}
}}

Text:
{chunks_text}"""

    def _parse_response(self, content: str) -> dict:
        content = re.sub(r"```(?:json)?\s*", "", content)
        content = content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"attributes": {name: _NULL_ENTRY.copy() for name in ATTRIBUTES}}
        
        attrs = parsed.get("attributes", {})
        for name in ATTRIBUTES:
            if name not in attrs:
                attrs[name] = _NULL_ENTRY.copy()
        return {"attributes": attrs}