from retrieval.baseline_rag import BaselineRAG
from agents.certification_agent import CertificationAgent
from agents.connectivity_agent import ConnectivityAgent
from agents.electrical_agent import ElectricalAgent
from agents.enviromental_agent import EnvironmentalAgent
from agents.physical_agent import PhysicalAgent

class Orchestrator:
    def __init__(self, index, metadata):
        self.certification_agent = CertificationAgent()
        self.connectivity_agent = ConnectivityAgent()
        self.electrical_agent = ElectricalAgent()
        self.environmental_agent = EnvironmentalAgent()
        self.physical_agent = PhysicalAgent()
        self.rag = BaselineRAG(index = index, metadata = metadata)

    def run(self, product_id):
        retrieved_chunks = self.rag.search("product technical specifications", product_id=product_id)
        chunks =[results.chunk_text for results in retrieved_chunks]
        certification_results = self.certification_agent.extract(chunks)
        connectivity_results = self.connectivity_agent.extract(chunks)
        electrical_results = self.electrical_agent.extract(chunks)
        environmental_results = self.environmental_agent.extract(chunks)
        physical_results = self.physical_agent.extract(chunks)

        merged_attributes = {
            **certification_results["attributes"],
            **connectivity_results["attributes"],
            **electrical_results["attributes"],
            **environmental_results["attributes"],
            **physical_results["attributes"],
        }
        return merged_attributes