from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import hashlib
import json
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

from src.utility import BBox


def generate_node_id(
    node: "BaseNode",
) -> str:
    """
    Generate a deterministic UUID representing the identity
    of an extracted node inside a document.

    Identity fields:
    - document_id
    - page_number
    - bbox (rounded)
    - node_type
    - image_hash (ImageNode only)
    """

    payload: dict[str, Any] = {
        "document_id": node.document_id,
        "page_number": node.page_number,
        "bbox": tuple(
            round(v, 2)
            for v in node.bbox
        ),
        "node_type": node.__class__.__name__,
    }

    image_hash = getattr(
        node,
        "image_hash",
        None,
    )

    if image_hash is not None:
        payload["image_hash"] = image_hash

    payload_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            payload_json,
        )
    )


class BaseNode(BaseModel, ABC):

    model_config = ConfigDict(
        extra="forbid",
        frozen="True",
    )

    document_id: str

    page_number: int = Field(
        ge=1,
    )

    # (x0, y0, x1, y1)
    bbox: BBox

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @computed_field
    @property
    def node_id(self) -> str:
        return generate_node_id(self)

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

    image_path: Path

    image_hash: str

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