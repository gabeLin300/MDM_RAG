import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class BaseAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
        )

    def extract(self, chunks: list[str]):
        prompt = self.get_prompt(chunks)
        response = self.llm.invoke(prompt)
        response_dict = json.loads(response.content)
        return response_dict  

    def get_prompt(self, chunks: list[str]) -> str:
        raise NotImplementedError("Subclasses must implement get_prompt method")