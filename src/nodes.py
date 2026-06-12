from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

class BaseNode(BaseModel, ABC):

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    node_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    document_id: str

    page_number: int = Field(
        ge=0,
    )

    # (x0, y0, x1, y1)
    bbox: tuple[
        float,
        float,
        float,
        float,
    ]

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x0 + self.x1) / 2,
            (self.y0 + self.y1) / 2,
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    @abstractmethod
    def content(self) -> str:
        """
        Canonical textual representation used by:
        - chunking
        - embedding
        - retrieval
        - indexing
        """
        pass


class TextNode(BaseNode):

    text: str

    @property
    def content(self) -> str:
        return self.text


class ImageNode(BaseNode):

    image_path: str | Path

    image_hash:str

    caption: str | None = None

    ocr_text: str | None = None

    @property
    def content(self) -> str:

        parts: list[str] = []

        if self.caption:
            parts.append(self.caption)

        if self.ocr_text:
            parts.append(self.ocr_text)

        return "\n".join(parts)


class TableNode(BaseNode):

    rows: list[list[Any]]

    caption: str | None = None

    @property
    def content(self) -> str:

        if self.caption:
            return self.caption

        return ""

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:

        if not self.rows:
            return 0

        return max(
            len(row)
            for row in self.rows
        )

def sort_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Natural reading order:
    page -> top -> left
    """

    return sorted(
        nodes,
        key=lambda node: (
            node.page_number,
            node.y0,
            node.x0,
        ),
    )