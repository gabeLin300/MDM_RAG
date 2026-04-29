import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class UnifiedAgent:
    """Single unified agent that extracts all attributes in one LLM call."""

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
        )

    def extract(self, chunks: list[str]) -> dict:
        """Extract all attributes from chunks in a single LLM call."""
        if not chunks:
            return {
                "attributes": {
                    "certifications": None,
                    "standards_compliance": None,
                    "regulatory_approvals": None,
                    "safety_certifications": None,
                    "environmental_certifications": None,
                    "industry_certifications": None,
                    "connectivity_interfaces": None,
                    "protocols": None,
                    "communication_standards": None,
                    "network_compatibility": None,
                    "voltage_rating": None,
                    "current_rating": None,
                    "power_consumption": None,
                    "electrical_specifications": None,
                    "frequency": None,
                    "operating_temperature": None,
                    "storage_temperature": None,
                    "humidity_range": None,
                    "environmental_conditions": None,
                    "altitude_rating": None,
                    "dimensions": None,
                    "weight": None,
                    "material": None,
                    "finish": None,
                    "mounting_type": None,
                }
            }

        prompt = self.get_prompt(chunks)
        response = self.llm.invoke(prompt)
        response_dict = json.loads(response.content)
        return response_dict

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You are a specialist in extracting product specifications from technical documents.

Extract the following attributes from the text below if present. Return ONLY a raw JSON object with no markdown, no code blocks, no explanation. If an attribute is not found, set its value to null.

Certifications (from documents):
- certifications: CE, UL, FCC, etc.
- standards_compliance: RoHS, REACH, etc.
- regulatory_approvals: ETL Listed, etc.
- safety_certifications: UL 61010-1, etc.
- environmental_certifications: Energy Star, etc.
- industry_certifications: ISA 95, etc.

Connectivity (interfaces and protocols):
- connectivity_interfaces: USB, Ethernet, Serial, etc.
- protocols: Modbus, TCP/IP, CAN, etc.
- communication_standards: IEC 60870-5-104, etc.
- network_compatibility: IPv4, IPv6, etc.

Electrical specifications:
- voltage_rating: e.g., 24V DC
- current_rating: e.g., 0.5A
- power_consumption: e.g., 12W
- electrical_specifications: Max current, etc.
- frequency: e.g., 50-60 Hz

Environmental conditions:
- operating_temperature: e.g., -10 to 50°C
- storage_temperature: e.g., -20 to 70°C
- humidity_range: e.g., 0-95% RH
- environmental_conditions: IP rating, etc.
- altitude_rating: e.g., up to 2000m

Physical attributes:
- dimensions: e.g., 100mm x 50mm x 25mm
- weight: e.g., 150g
- material: e.g., aluminum, steel
- finish: e.g., powder coated
- mounting_type: e.g., DIN rail, panel mount

Expected format:
{{
    "attributes": {{
        "certifications": null,
        "standards_compliance": null,
        "regulatory_approvals": null,
        "safety_certifications": null,
        "environmental_certifications": null,
        "industry_certifications": null,
        "connectivity_interfaces": null,
        "protocols": null,
        "communication_standards": null,
        "network_compatibility": null,
        "voltage_rating": null,
        "current_rating": null,
        "power_consumption": null,
        "electrical_specifications": null,
        "frequency": null,
        "operating_temperature": null,
        "storage_temperature": null,
        "humidity_range": null,
        "environmental_conditions": null,
        "altitude_rating": null,
        "dimensions": null,
        "weight": null,
        "material": null,
        "finish": null,
        "mounting_type": null
    }}
}}

Text:
{chunks_text}"""
