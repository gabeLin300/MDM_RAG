from agents.base_agent import BaseAgent

class PhysicalAgent(BaseAgent):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        super().__init__(model_name)

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You are a specialist in extracting physical specifications from product documents.

        Extract the following physical attributes from the text below if present:
        - dimensions
        - weight
        - material
        - housing
        - mounting_type
        - enclosure_type

        Return ONLY a raw JSON object with no markdown, no code blocks, no explanation. If an attribute is not found, set its value to null.

        Expected format:
        {{
            "attributes": {{
                "dimensions": "150 x 80 x 45 mm",
                "weight": "320g",
                "material": "polycarbonate",
                "housing": "DIN rail",
                "mounting_type": "panel mount",
                "enclosure_type": "IP54"
            }}
        }}
        
        Text:
        {chunks_text}"""