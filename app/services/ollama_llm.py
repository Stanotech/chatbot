import requests
from langchain.llms.base import LLM
from pydantic import Field


class OllamaLLM(LLM):
    ollama_url: str = Field(...)

    def _call(self, prompt: str, stop: list = []) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "mistral-finance-ft", "prompt": prompt, "stream": False},
        )
        return resp.json().get("response", "")

    @property
    def _identifying_params(self) -> dict:
        return {"ollama_url": self.ollama_url}

    @property
    def _llm_type(self) -> str:
        return "ollama"
