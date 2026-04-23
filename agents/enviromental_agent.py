from agents.base_agent import BaseAgent

class EnvironmentalAgent(BaseAgent):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        super().__init__(model_name)

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You are a specialist in extracting environmental specifications from product documents.

        Extract the following environmental attributes from the text below if present:
        - operating_temperature
        - storage_temperature
        - humidity
        - ingress_protection
        - shock_resistance
        - vibration_resistance

        Return ONLY a raw JSON object with no markdown, no code blocks, no explanation. If an attribute is not found, set its value to null.

        Expected format:
        {{
            "attributes": {{
                "operating_temperature": "-20 to 60°C",
                "storage_temperature": "-40 to 85°C",
                "humidity": "5% to 95% non-condensing",
                "ingress_protection": "IP65",
                "shock_resistance": "15g for 11ms",
                "vibration_resistance": "2g for 10-500Hz"
            }}
        }}

        Text:
        {chunks_text}"""