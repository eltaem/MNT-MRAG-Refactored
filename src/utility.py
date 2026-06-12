import re
import hashlib
from pathlib import Path
import uuid
from typing import TypeAlias
import fitz

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

def generate_document_id(file_path: str | Path,) -> str:
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    content_hash = sha256.hexdigest()

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            content_hash,
        )
    )

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