from __future__ import annotations
from enum import IntEnum, StrEnum
import gc
import json
from json_repair import repair_json
import re
import hashlib
from pathlib import Path
import uuid
from typing import Any, Literal, Protocol, Sequence, TypeAlias, TypeVar
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

# ==========================================================
# LOGGING SETTINGS
# ==========================================================

configure_logging(LogLevel.INFO)

T = TypeVar("T", bound=BaseModel)

# ==========================================================
# Single model
# ==========================================================

def save_model(model: BaseModel, path: str | Path) -> Path:
    """
    Save a single Pydantic model to a JSON file.

    Returns
    -------
    Path
        The path of the saved file.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        model.model_dump_json(indent=4),
        encoding="utf-8",
    )

    return path


def load_model(model_cls: type[T], path: str | Path) -> T:
    """
    Load a single Pydantic model from a JSON file.

    Returns
    -------
    T
        An instance of the specified Pydantic model subclass.
    """
    path = Path(path)

    return model_cls.model_validate_json(
        path.read_text(encoding="utf-8")
    )

def save_models(
    models: Sequence[BaseModel],
    path: str | Path,
    *,
    mode: Literal["single", "directory"] = "single",
) -> Path:
    """
    Save multiple Pydantic models.

    Parameters
    ----------
    models
        Models to save.
    path
        Output file (single mode) or output directory (directory mode).
    mode
        - "single": save all models into one JSON array.
        - "directory": save each model into its own JSON file.

    Returns
    -------
    Path
        The file or directory written to.
    """
    path = Path(path)

    match mode:
        case "single":
            if path.exists() and path.is_dir():
                raise ValueError(
                    f"Expected a file path, got directory: {path}"
                )

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            payload = [
                model.model_dump(mode="json")
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

        case "directory":
            if path.exists() and path.is_file():
                raise ValueError(
                    f"Expected a directory path, got file: {path}"
                )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            for index, model in enumerate(models):
                save_model(
                    model,
                    path / f"{index}.json",
                )

    return path


def load_models(
    model_cls: type[T],
    path: str | Path,
) -> list[T]:
    """
    Load multiple Pydantic models.

    If `path` is a JSON file, it is assumed to contain a JSON array of models.

    If `path` is a directory, every '*.json' file inside it is loaded.

    Returns
    -------
    list[T]
        Loaded model instances.
    """
    path = Path(path)

    if not path.exists():
        return []

    if path.is_file():
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(payload, list):
            raise ValueError(
                "Expected a JSON array when loading multiple models."
            )

        return [
            model_cls.model_validate(item)
            for item in payload
        ]

    json_files = sorted(
        path.glob("*.json"),
        key=lambda p: (
            int(p.stem)
            if p.stem.isdigit()
            else p.stem
        ),
    )

    return [
        load_model(model_cls, file_path)
        for file_path in json_files
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

def extract_json_object(
    text: str,
) -> dict[str, Any]:

    text = text.strip()

    # Remove Markdown fences
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        ).strip()

    # -----------------------------------------------------
    # Strict parsing
    # -----------------------------------------------------

    start = text.find("{")

    if start != -1:

        try:
            obj, _ = json.JSONDecoder().raw_decode(
                text[start:]
            )

            if isinstance(obj, dict):
                return obj

        except json.JSONDecodeError:
            pass

    # -----------------------------------------------------
    # Automatic repair
    # -----------------------------------------------------

    try:
        repaired = repair_json(
            text,
            return_objects=True,
        )

        if isinstance(repaired, dict):

            logger.warning(
                "Malformed JSON output was automatically repaired."
            )

            return repaired

    except Exception as e:

        logger.warning(
            "JSON repair failed: %s",
            e,
        )

    raise ValueError(
        "No valid JSON object found in model output.",
    )

class Modality(StrEnum):
    TEXT = "text"
    TEXT_TABLE = "text_table"
    TEXT_IMAGE = "text_image"
    TEXT_TABLE_IMAGE = "text_table_image"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " + ")

def escape_md(text: str) -> str:
    """Escape markdown-sensitive characters in raw text."""
    # Escape backticks, pipes, and newlines for code blocks
    return text.replace("`", "\\`").replace("|", "\\|")