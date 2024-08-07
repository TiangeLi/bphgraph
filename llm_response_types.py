from typing_extensions import TypedDict
from typing import List

class BooleanResponse(TypedDict):
    b: bool

class StringResponse(TypedDict):
    s: str

class ListOfStringsResponse(TypedDict):
    l: List[str]