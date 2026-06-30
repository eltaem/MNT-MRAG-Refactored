from __future__ import annotations
from pathlib import Path
from typing import Self
import json

from src.configs import (
    BaseExtractorConfig,
    TextExtractorConfig,
    ImageExtractorConfig,
    TableExtractorConfig,
    PDFExtractorConfig
)

from src.nodes import (
    BaseNode,
    TextNode,
    ImageNode,
    TableNode
)
import hashlib
import fitz
from src.utility import (
    image_xref_to_bbox,
    BBox,
    EMPTY_BBOX,
    sort_nodes,
    generate_document_id
)
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    def __init__(self, config: BaseExtractorConfig):
        self.config = config

    @abstractmethod
    def extract(self, page: fitz.Page, document_id: str):
        pass

class TextExtractor(BaseExtractor):

    def __init__(self, config: TextExtractorConfig,):
        super().__init__(config)
        self.config = config

    def extract(self, page: fitz.Page, document_id: str,) -> list[TextNode]:
        text_nodes: list[TextNode] = []

        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks",[],):
            # 0 = text block
            if block.get("type") != 0:
                continue

            text = self._extract_block_text(block)

            if self.config.skip_empty_blocks and not text:
                continue

            if len(text) < self.config.min_text_length:
                continue

            text_nodes.append(
                TextNode(
                    document_id=document_id,
                    page_number=page.number + 1,
                    bbox=self._block_bbox(block),
                    text=text,
                    metadata={
                        "block_number": block.get(
                            "number"
                        ),
                        "line_count": len(
                            block.get(
                                "lines",
                                [],
                            )
                        ),
                    },
                )
            )

        return text_nodes

    def _extract_block_text(self, block: dict,) -> str:

        lines: list[str] = []

        for line in block.get("lines", [],):

            spans: list[str] = []

            for span in line.get(
                "spans",
                [],
            ):

                text = span.get(
                    "text",
                    ""
                )

                if self.config.strip_whitespace:
                    text = text.strip()

                if text:
                    spans.append(
                        text
                    )

            if spans:

                lines.append(
                    " ".join(
                        spans
                    )
                )

        return "\n".join(
            lines
        )

    def _block_bbox(
        self,
        block: dict,
    ) -> BBox:

        bbox = block.get(
            "bbox"
        )

        if bbox is None:
            return EMPTY_BBOX

        return (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )

class ImageExtractor(BaseExtractor):
    """
    Extract raster images embedded inside a PDF page.

    Features
    --------
    - Reading-order sorting (top -> left)
    - Runtime deduplication
    - Idempotent filesystem writes
    - Human-readable filenames:
        image_001_87828e1b.png
    - Stable image hashes for downstream RAG
    """

    def __init__(
        self,
        config: ImageExtractorConfig,
    ):
        super().__init__(config)

        self.config = config

        # Runtime optimization only.
        # Not persisted.
        self._seen_hashes: set[str] = set()

        self.config.image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def extract(
        self,
        page: fitz.Page,
        document_id: str,
    ) -> list[ImageNode]:

        images = page.get_images(
            full=True,
        )

        if not images:
            return []

        page_dir = (
            self.config.image_dir
            / document_id
            / f"page_{page.number + 1:03d}"
        )

        page_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidates: list[dict] = []

        for image_info in images:

            xref = image_info[0]

            image_dict = self._extract_image(
                page,
                xref,
            )

            if image_dict is None:
                continue

            image_bytes = image_dict.get(
                "image"
            )

            if image_bytes is None:
                continue

            width = image_dict.get(
                "width",
                0,
            )

            height = image_dict.get(
                "height",
                0,
            )

            if not self._passes_size_filter(
                width,
                height,
            ):
                continue

            image_hash = self._compute_hash(
                image_bytes
            )

            if self._is_duplicate(
                image_hash
            ):
                continue

            candidates.append(
                {
                    "xref": xref,
                    "image_bytes": image_bytes,
                    "image_hash": image_hash,
                    "short_hash":
                        image_hash[:8],
                    "extension":
                        image_dict.get(
                            "ext",
                            "png",
                        ),
                    "width": width,
                    "height": height,
                    "bbox":
                        image_xref_to_bbox(
                            page,
                            xref,
                        ),
                }
            )

        # Natural reading order
        candidates.sort(
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            )
        )

        image_nodes: list[ImageNode] = []

        for idx, item in enumerate(
            candidates,
            start=1,
        ):

            image_path = (
                page_dir
                / (
                    f"image_"
                    f"{idx:03d}_"
                    f"{item['short_hash']}."
                    f"{item['extension']}"
                )
            )

            # Idempotent extraction
            if not image_path.exists():
                image_path.write_bytes(
                    item["image_bytes"]
                )

            image_nodes.append(
                ImageNode(
                    document_id=document_id,
                    page_number=page.number + 1,
                    bbox=item["bbox"],
                    image_path=image_path,
                    image_hash=item[
                        "image_hash"
                    ],
                    metadata={
                        "xref":
                            item["xref"],
                        "width":
                            item["width"],
                        "height":
                            item["height"],
                        "extension":
                            item[
                                "extension"
                            ],
                    },
                )
            )

        return image_nodes

    def reset_cache(
        self,
    ) -> None:
        """
        Clear runtime deduplication cache.

        Useful when processing a completely
        separate batch of documents.
        """

        self._seen_hashes.clear()

    def _extract_image(
        self,
        page: fitz.Page,
        xref: int,
    ) -> dict | None:

        try:
            return page.parent.extract_image(
                xref
            )

        except Exception:
            return None

    def _passes_size_filter(
        self,
        width: int,
        height: int,
    ) -> bool:

        return (
            width >= self.config.min_width
            and height >= self.config.min_height
        )

    def _is_duplicate(
        self,
        image_hash: str,
    ) -> bool:

        if not self.config.deduplicate:
            return False

        if image_hash in self._seen_hashes:
            return True

        self._seen_hashes.add(
            image_hash
        )

        return False

    @staticmethod
    def _compute_hash(
        image_bytes: bytes,
    ) -> str:

        return hashlib.sha256(
            image_bytes
        ).hexdigest()

class TableExtractor(
    BaseExtractor
):

    def __init__(
        self,
        config: TableExtractorConfig,
    ):
        super().__init__(config)

        self.config = config

    def extract(
        self,
        page: fitz.Page,
        document_id: str,
    ) -> list[TableNode]:

        table_nodes: list[TableNode] = []

        finder = page.find_tables()

        for table_index, table in enumerate(finder.tables):

            rows = table.extract()

            if not rows:
                continue

            rows = self._normalize_rows(rows)

            if self.config.skip_empty_tables and not rows:
                continue

            if len(rows) < self.config.min_rows:
                continue

            if self._column_count(rows) < self.config.min_columns:
                continue

            table_nodes.append(
                TableNode(
                    document_id=document_id,
                    page_number=page.number + 1,
                    bbox=self._table_bbox(
                        table.bbox
                    ),
                    rows=rows,
                    metadata={
                        "table_index":
                            table_index,
                    },
                )
            )

        return table_nodes

    def _normalize_rows(self, rows: list[list],) -> list[list]:

        normalized_rows = []

        for row in rows:
            normalized_rows.append(
                ["" if cell is None else str(cell) for cell in row]
            )

        return normalized_rows

    def _column_count(self, rows: list[list],) -> int:
        if not rows:
            return 0

        return max(len(row) for row in rows)

    def _table_bbox(self, bbox,) -> BBox:

        return (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )

class PDFExtractor:
    """
    Orchestrates extraction of:
    - text nodes
    - image nodes
    - table nodes

    Responsibilities:
    - open PDF
    - invoke page extractors
    - aggregate nodes
    - sort nodes into reading order

    Non-responsibilities:
    - captioning
    - OCR
    - chunking
    - embedding
    - indexing
    - enrichment
    """

    def __init__(self,config: PDFExtractorConfig):
        self.config = config

        self.text_extractor = TextExtractor(config.text)

        self.image_extractor = ImageExtractor(config.image)

        self.table_extractor = TableExtractor(config.table)

    @classmethod
    def from_config(cls, config: PDFExtractorConfig,) -> Self:
        return cls(config)

    def extract(self, path: str | Path,) -> list[BaseNode]:

        path = Path(path)

        document = fitz.open(path)

        document_id = generate_document_id(path)

        nodes: list[BaseNode] = []

        try:
            for page in document:

                nodes.extend(
                    self.text_extractor.extract(
                        page,
                        document_id,
                    )
                )

                nodes.extend(
                    self.image_extractor.extract(
                        page,
                        document_id,
                    )
                )

                nodes.extend(
                    self.table_extractor.extract(
                        page,
                        document_id,
                    )
                )

        finally:
            document.close()

        return sort_nodes(nodes)

    def extract_many(self, directory: str | Path, pattern: str = "*.pdf") -> dict[Path, list[BaseNode]]:

        directory = Path(directory)

        documents: dict[Path, list[BaseNode],] = {}

        for pdf_path in sorted(directory.rglob(pattern)):
            documents[pdf_path] = self.extract(pdf_path)

        return documents

    @staticmethod
    def save_json(nodes: list[BaseNode], path: str | Path,) -> Path:

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        data = [node.model_dump(mode="json") for node in nodes]

        path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8",)

        return path