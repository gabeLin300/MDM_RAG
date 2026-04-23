from agents.base_agent import BaseAgent

class ElectricalAgent(BaseAgent):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        super().__init__(model_name)

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You are a specialist in extracting electrical specifications from product documents.

        Extract the following electrical attributes from the text below if present:
        - voltage
        - current
        - power
        - frequency
        - power_supply
        - power_consumption

        Return ONLY a raw JSON object with no markdown, no code blocks, no explanation. If an attribute is not found, set its value to null.

        Expected format:
        {{
            "attributes": {{
                "voltage": "24V DC",
                "current": "0.5A",
                "power": "12W",
                "frequency": "50/60 Hz",
                "power_supply": "AC",
                "power_consumption": null
           }}
        }}

        Text:
        {chunks_text}"""