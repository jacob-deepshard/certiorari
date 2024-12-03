from typing import Callable, Dict, List, Optional, TypedDict

from pydantic import BaseModel

class ChatMessage(TypedDict):
    role: str
    content: str

class LLM:
    def __init__(self): ...
    def chat(self, messages: List[ChatMessage]) -> str: ...
    def generate_text(self, prompt: str) -> str: ...
    def embed_text(self, text: str) -> List[float]: ...
    def constrained_generate_text(self, prompt: str, ebnf: str) -> str: ...
    def structured_output(self, schema: type[BaseModel]) -> BaseModel: ...

class VectorStore:
    def __init__(self): ...
    def add_text(self, text: str, metadata: Dict[str, str]) -> None: ...
    def get_text(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.5,
        filter: Optional[
            Callable[[str, Dict[str, str]], bool]
        ] = None,  # filter by metadata, return True to include
        sort: Dict[
            str, str
        ] = {},  # sort by metadata, key is field, value is "asc" or "desc"
    ) -> List[str]: ...
