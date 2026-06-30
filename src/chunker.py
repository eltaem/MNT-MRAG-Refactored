from abc import ABC, abstractmethod
import asyncio
import json
from pathlib import Path
import uuid

from llama_index.core import Document
from pydantic import BaseModel, Field

from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser
from src.configs import BaseChunkerConfig, DoublePassMergingChunkerConfig
from src.embedder import Embedder
from src.nodes import BaseNode
from src.utility import load_models, logger, save_models

class Chunk(BaseModel):
    """
    Smallest retrievable unit.

    Produced by a Chunker.
    """

    chunk_id: str

    document_id: str

    node_type: str

    text: str

    metadata: dict = Field(
        default_factory=dict,
    )


class BaseChunker(ABC):

    def __init__(
        self,
        config: BaseChunkerConfig,
    ):
        self.config = config

    @abstractmethod
    def chunk(
        self,
        nodes: list[BaseNode],
    ) -> list[Chunk]:
        ...

    
    async def achunk(
        self,
        nodes: list[BaseNode],
    ) -> list[Chunk]:
        return await asyncio.to_thread(
            self.chunk,
            nodes,
        )
    
    @staticmethod
    def to_json(
        chunks: list[Chunk],
        path: str | Path,
    ) -> Path:

        return save_models(
            chunks,
            path,
        )
    
    @staticmethod
    def from_json(
        path: str | Path,
    ) -> list[Chunk]:

        return load_models(
            path,
            Chunk,
        )

class DoublePassMergingChunker(
    BaseChunker,
):

    def __init__(
        self,
        config: DoublePassMergingChunkerConfig,
        embedder: Embedder,
    ):
        super().__init__(config)

        self.config = config
        self.embedder = embedder

        self._splitter = (
            SemanticDoubleMergingSplitterNodeParser(
                embed_model=self.embedder,
                initial_threshold=(
                    config.initial_threshold
                ),
                appending_threshold=(
                    config.appending_threshold
                ),
                merging_threshold=(
                    config.merging_threshold
                ),
                max_chunk_size=(
                    config.chunk_size
                ),
            )
        )

    def chunk(
        self,
        nodes: list[BaseNode],
    ) -> list[Chunk]:

        logger.info(
            "Chunking %d nodes using %s...",
            len(nodes),
            self.__class__.__name__,
        )

        chunks: list[Chunk] = []

        for node in nodes:

            text = node.content.strip()

            if not text:
                logger.debug(
                    "Skipping %s (%s): empty content.",
                    node.node_type,
                    node.node_id,
                )
                continue

            llama_document = Document(
                text=text,
            )

            llama_nodes = (
                self._splitter.get_nodes_from_documents(
                    [llama_document]
                )
            )

            logger.debug(
                "Node %s (%s) -> %d chunks.",
                node.node_id,
                node.node_type,
                len(llama_nodes),
            )

            for idx, llama_node in enumerate(
                llama_nodes
            ):

                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        f"{node.node_id}::{idx}",
                    )
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=node.document_id,
                        node_id=node.node_id,
                        node_type=node.node_type,
                        text=llama_node.text,
                        metadata={
                            "page_number": node.page_number,
                            "chunk_index": idx,
                        },
                    )
                )

                logger.debug(
                    "Created chunk %s (%d chars).",
                    chunk_id,
                    len(llama_node.text),
                )

        logger.info(
            "Finished chunking: %d nodes -> %d chunks.",
            len(nodes),
            len(chunks),
        )

        return chunks