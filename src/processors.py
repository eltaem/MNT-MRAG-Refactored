from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
from pathlib import Path

from src.enrichers import BaseNodeEnricher
from src.extractors import PDFExtractor
from src.nodes import NODE_REGISTRY, BaseNode
from src.configs import (
    BaseNodeProcessorConfig,
    PDFProcessorConfig,
    RemoveBeforePatternProcessorConfig,
    TextMergerConfig,
    TextMergeMode
)
from src.nodes import TextNode
import re
from src.utility import collect_pdfs, load_json, save_models

class BaseNodeProcessor(ABC):
    """
    Transform:

        list[BaseNode]
            ↓
        list[BaseNode]
    """

    def __init__(
        self,
        config: BaseNodeProcessorConfig,
    ):
        self.config = config

    @abstractmethod
    def process(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:
        pass

    async def aprocess(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:

        return await asyncio.to_thread(
            self.process,
            nodes,
        )

class RemoveBeforePatternProcessor(BaseNodeProcessor):
    def __init__(
        self,
        config:
        RemoveBeforePatternProcessorConfig,
    ):
        super().__init__(config)

        self.config = config

        # Normalize patterns once
        self._patterns = [self._normalize(pattern) for pattern in self.config.patterns]

    def process(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:

        if not self._patterns:
            return nodes

        boundary_idx = self._find_boundary(
            nodes
        )

        if boundary_idx is None:
            return nodes

        return nodes[boundary_idx:]

    def _find_boundary(self,nodes: list[BaseNode],) -> int | None:

        for idx, node in enumerate(nodes):
            if not isinstance(node, TextNode,):
                continue

            text = self._normalize(node.text)

            for pattern in self._patterns:

                if (
                    self.config.startswith
                ):
                    matched = (
                        text.startswith(
                            pattern
                        )
                    )
                else:
                    matched = (
                        pattern in text
                    )

                if matched:
                    return idx

        return None

    def _normalize(self, text: str) -> str:

        if self.config.ignore_case:
            text = text.upper()

        if self.config.normalize_punctuation:
            text = re.sub(r"[.:]+", " ", text)

        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)

        return text.strip()

class TextMerger(BaseNodeProcessor):
    def __init__(self, config: TextMergerConfig,):
        super().__init__(config)
        self.config = config

    def process(self, nodes: list[BaseNode],) -> list[BaseNode]:

        if not nodes:
            return []

        result: list[BaseNode] = []
        buffer: list[TextNode] = []

        def flush() -> None:
            if not buffer:
                return

            result.append(self._merge_nodes(buffer))

            buffer.clear()

        for node in nodes:

            if isinstance(node, TextNode):

                if (
                    self.config.mode == TextMergeMode.PAGE
                    and buffer
                    and (node.page_number != buffer[-1] .page_number)
                ):
                    flush()

                buffer.append(node)
                continue

            #
            # ImageNode / TableNode
            # become hard boundaries
            #
            flush()
            result.append(node)

        flush()

        return result

    def _merge_nodes(self, nodes: list[TextNode],) -> TextNode:

        if len(nodes) == 1:
            return nodes[0]

        first = nodes[0]

        text = self.config.separator.join(node.text for node in nodes)

        x0 = min(
            node.x0
            for node in nodes
        )
        y0 = min(
            node.y0
            for node in nodes
        )
        x1 = max(
            node.x1
            for node in nodes
        )
        y1 = max(
            node.y1
            for node in nodes
        )

        metadata = {}

        if self.config.merge_metadata:
            metadata = {
                "merged_node_ids": [
                    node.node_id
                    for node in nodes
                ],
                "merged_count": len(
                    nodes
                ),
            }

        return TextNode(
            document_id=(
                first.document_id
            ),
            page_number=(
                first.page_number
            ),
            bbox=(
                x0,
                y0,
                x1,
                y1,
            ),
            text=text,
            metadata=metadata,
        )

class PDFProcessor:
    """
    High-level PDF processing orchestrator.

        PDF
         ↓
        PDFExtractor
         ↓
        list[BaseNode]
         ↓
        Processors
         ↓
        Enrichers
    """

    def __init__(
        self,
        config:PDFProcessorConfig,
        extractor: PDFExtractor,
        processors: list[BaseNodeProcessor] | None = None,
        enrichers: list[BaseNodeEnricher] | None = None,
        ):
        
        self.config = config
        self.extractor = extractor

        self.processors = (
            processors
            if processors is not None
            else []
        )

        self.enrichers = (
            enrichers
            if enrichers is not None
            else []
        )

    def process_pdf(self, path: str | Path,) -> list[BaseNode]:

        nodes = self.extractor.extract(path)

        for processor in self.processors:
            nodes = processor.process(nodes)

        for enricher in self.enrichers:
            nodes = enricher.enrich(nodes)

        return nodes

    def process_pdfs(self, paths: list[str | Path],) -> list[BaseNode]:

        nodes: list[BaseNode] = []

        for path in paths:
            nodes.extend(self.process_pdf(path))

        return nodes
    
    def process_pdfs_dir(self, directory: str | Path,) -> list[BaseNode]:

        pdf_paths = collect_pdfs(directory)

        return self.process_pdfs(pdf_paths)

    async def aprocess_pdf(self, path: str | Path,) -> list[BaseNode]:

        nodes = self.extractor.extract(path)

        for processor in (
            self.processors
        ):
            nodes = await processor.aprocess(nodes)

        for enricher in self.enrichers:
            nodes = await enricher.aenrich(nodes)

        return nodes

    async def aprocess_pdfs(
        self,
        paths: list[str | Path]
    ) -> list[BaseNode]:
        """
        Process multiple PDFs with controlled concurrency.

        batch_size controls how many PDFs are processed concurrently
        to avoid LLM / GPU overload.
        """

        if not paths:
            return []

        semaphore = asyncio.Semaphore(self.config.pdf_concurrency)

        async def run_one(path: str | Path) -> list[BaseNode]:
            async with semaphore:
                return await self.aprocess_pdf(path)

        # run with controlled concurrency
        tasks = [run_one(p) for p in paths]

        results = await asyncio.gather(*tasks)

        # flatten
        nodes: list[BaseNode] = []
        for batch in results:
            nodes.extend(batch)

        return nodes
    
    async def aprocess_pdfs_dir(
        self,
        directory: str | Path,
    ) -> list[BaseNode]:

        pdf_paths = collect_pdfs(
            directory
        )

        return await self.aprocess_pdfs(
            pdf_paths
        )

    @staticmethod
    def to_json(
        nodes: list[BaseNode],
        path: str | Path,
    ) -> Path:

        return save_models(
            nodes,
            path,
        )
    
    @staticmethod
    def from_json(
        path: str | Path,
    ) -> list[BaseNode]:

        payload = load_json(path)

        nodes: list[BaseNode] = []

        for item in payload:

            node_type = item.pop(
                "node_type"
            )
            item.pop("node_id", None)

            node_cls = NODE_REGISTRY[node_type]

            node = node_cls.model_validate(item)
            nodes.append(node)

        return nodes