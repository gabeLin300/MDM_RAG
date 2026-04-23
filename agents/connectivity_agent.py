from agents.base_agent import BaseAgent

class ConnectivityAgent(BaseAgent):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        super().__init__(model_name)

    def get_prompt(self, chunks: list[str]) -> str:
        chunks_text = "\n\n".join(chunks)
        return f"""You are a specialist in extracting connectivity specifications from product documents.

        Extract the following connectivity attributes from the text below if present:
        - communication_protocols
        - wired_interfaces
        - ports
        - network_capabilities
        - data_rate
        - bus_type

        Return ONLY a raw JSON object with no markdown, no code blocks, no explanation. If an attribute is not found, set its value to null.

        Expected format:
        {{
            "attributes": {{
                "communication_protocols": "BACnet, Modbus, M-Bus, MSTP",
                "wired_interfaces": "Ethernet, RS-485",
                "ports": "3x Ethernet, 3x RS-485",
                "network_capabilities": "IPv4",
                "data_rate": "100 Mbps",
                "bus_type": "T1L"
            }}
        }}

        Text:
        {chunks_text}"""