from typing import TypedDict


class CreateDocument(TypedDict):
    text: str
    metadata: dict[str, str] | None


class ResultDocument(TypedDict):
    text: str
    metadata: dict[str, str] | None
    pk: int


class SearchResult(TypedDict):
    id: int
    distance: float
    entity: ResultDocument