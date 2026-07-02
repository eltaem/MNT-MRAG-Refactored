from __future__ import annotations
from enum import IntEnum
import gc
import json
import re
import hashlib
from pathlib import Path
import uuid
from typing import Any, Protocol, TypeAlias, TypeVar
import fitz
from typing import TYPE_CHECKING
import logging

from pydantic import BaseModel
import torch

if TYPE_CHECKING:
    from src.nodes import BaseNode

def clean_text(text: str) -> str:
    """
    Clean generated text by removing extra whitespace.
    
    Args:
        text: Raw text from captioning model
        
    Returns:
        Cleaned text
    """
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text

def deterministic_uuid(value: Any) -> str:
    """
    Generate deterministic UUID5 from any JSON-serializable value.
    """

    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            payload,
        )
    )

def generate_document_id(file_path: str | Path,) -> str:
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return deterministic_uuid(sha256.hexdigest())

BBox: TypeAlias = tuple[
    float,
    float,
    float,
    float,
]

EMPTY_BBOX: BBox = (
    0.0,
    0.0,
    0.0,
    0.0,
)

def rect_to_bbox(rect: fitz.Rect,) -> BBox:
    return (
        rect.x0,
        rect.y0,
        rect.x1,
        rect.y1,
    )


def image_xref_to_bbox(page: fitz.Page, xref: int) -> BBox:

    try:
        rects = page.get_image_rects(xref)

    except Exception:
        return EMPTY_BBOX

    if not rects:
        return EMPTY_BBOX

    return rect_to_bbox(rects[0])

def width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def height(bbox: BBox) -> float:
    return bbox[3] - bbox[1]


def area(bbox: BBox) -> float:
    return width(bbox) * height(bbox)

def center(bbox: BBox,) -> tuple[float, float]:
    return (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
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

def collect_pdfs(directory: str | Path) -> list[Path]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(directory.glob("*.pdf"), key=lambda p: p.name)

def serialize_table(table: Any) -> str:
    if isinstance(table, str):
        return table

    if not isinstance(table, list):
        return str(table)

    if not table:
        return ""

    lines: list[str] = []

    # Header
    header = [
        "" if c is None else str(c)
        for c in table[0]
    ]
    lines.append(" | ".join(header))
    lines.append(" | ".join("---" for _ in header))

    # Rows
    for row in table[1:]:
        lines.append(
            " | ".join(
                "" if c is None else str(c)
                for c in row
            )
        )

    return "\n".join(lines)

logger = logging.getLogger("MNT_MRAG")

class LogLevel(IntEnum):
    SILENT = logging.CRITICAL + 1
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

def configure_logging(level: LogLevel):
    logging.basicConfig(
        level=int(level),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )
    logger.setLevel(int(level))

configure_logging(LogLevel.INFO)

T = TypeVar("T", bound=BaseModel)

# ==========================================================
# Single model
# ==========================================================

def save_model(
    model: BaseModel,
    path: str | Path,
) -> Path:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            model.model_dump(
                mode="json",
            ),
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def load_model(
    path: str | Path,
    model_cls: type[T],
) -> T:

    path = Path(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    return model_cls.model_validate(
        payload,
    )

def save_models(
    models: list[BaseModel],
    path: str | Path,
) -> Path:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = [
        model.model_dump(
            mode="json",
        )
        for model in models
    ]

    path.write_text(
        json.dumps(
            payload,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def load_models(
    path: str | Path,
    model_cls: type[T],
) -> list[T]:

    path = Path(path)

    payload = load_json(path)

    return [
        model_cls.model_validate(item)
        for item in payload
    ]

def load_json(
    path: str | Path,
) -> Any:

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

class HasName(Protocol):
    @property
    def name(self) -> str: ...

def generate_collection_name(
    prefix: str,
    *components: HasName,
) -> str:
    parts = [prefix]

    for component in components:
        parts.append(component.name)

    return "-".join(parts)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def sanitize_name(value: str) -> str:
    return (
        value
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )