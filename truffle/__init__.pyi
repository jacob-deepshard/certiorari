from dataclasses import dataclass
from typing import Dict, List, TypedDict

from pydantic import BaseModel

@dataclass
class AppMetadata:
    name: str
    fullname: str
    description: str
    goal: str

def tool(func): ...
