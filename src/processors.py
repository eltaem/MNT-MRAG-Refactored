from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from src.nodes import BaseNode
from src.configs import (
    BaseNodeProcessorConfig,
    RemoveBeforePatternProcessorConfig,
    TextMergerConfig,
    TextMergeMode
)
from src.nodes import TextNode
import re

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