from agents.base_agent import BaseAgent

class CertificationAgent(BaseAgent):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        super().__init__(model_name)

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You are a specialist in extracting certification specifications from product documents.

        Extract the following certification attributes from the text below if present:
        - certifications
        - standards_compliance
        - regulatory_approvals
        - safety_certifications
        - environmental_certifications
        - industry_certifications

        Return ONLY a raw JSON object with no markdown, no code blocks, no explanation. If an attribute is not found, set its value to null.

        Expected format:
        {{
            "attributes": {{
                "certifications": "CE, UL, FCC",
                "standards_compliance": "RoHS, REACH",
                "regulatory_approvals": "ETL Listed",
                "safety_certifications": "UL 61010-1",
                "environmental_certifications": "Energy Star",
                "industry_certifications": "ISA 95"
            }}
        }}

        Text:
        {chunks_text}"""