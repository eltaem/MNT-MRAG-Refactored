from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

from llama_index.core import Document
from pydantic import BaseModel, Field

from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser
from src.configs import BaseChunkerConfig, DoublePassMergingChunkerConfig
from src.embedder import Embedder
from src.nodes import BaseNode
from src.utility import deterministic_uuid, load_models, logger, save_models

def generate_chunk_id(
    node_id: str,
    chunk_index: int,
) -> str:

    return deterministic_uuid({
        "node_id": node_id,
        "chunk_index": chunk_index,
    })

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

class ChunkEmbedding(BaseModel):

    chunk_id: str

    embedding: list[float]

class ChunkRecord(BaseModel):

    chunk: Chunk

    embedding: ChunkEmbedding

    @property
    def chunk_id(self):
        return self.chunk.chunk_id

    @property
    def text(self):
        return self.chunk.text

    @property
    def document_id(self):
        return self.chunk.document_id
    
    @classmethod
    def from_embeddings(
        cls,
        chunks: list[Chunk],
        embeddings: list[ChunkEmbedding],
    ) -> list["ChunkRecord"]:

        embedding_map = {
            emb.chunk_id: emb
            for emb in embeddings
        }

        missing = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.chunk_id not in embedding_map
        ]

        if missing:
            raise ValueError(
                f"Missing embeddings for {len(missing)} chunks."
            )

        return [
            cls(
                chunk=chunk,
                embedding=embedding_map[chunk.chunk_id],
            )
            for chunk in chunks
        ]

class BaseChunker(ABC):

    def __init__(
        self,
        config: BaseChunkerConfig,
    ):
        self.config = config
    
    @property
    def name(self) -> str:
        return self.config.name

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

class BaseSemanticChunker(ABC):

    def __init__(
        self,
        config: BaseChunkerConfig,
        embedder: Embedder,
    ):
        self.config = config
        self.embedder = embedder
    
    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def chunk(
        self,
        nodes: list[BaseNode],
    ) -> list[ChunkRecord]:
        ...

    async def achunk(
        self,
        nodes: list[BaseNode],
    ) -> list[ChunkRecord]:

        return await asyncio.to_thread(
            self.chunk,
            nodes,
        )

    @staticmethod
    def to_json(
        records: list[ChunkRecord],
        path: str | Path,
    ) -> Path:

        return save_models(
            records,
            path,
        )

    @staticmethod
    def from_json(
        path: str | Path,
    ) -> list[ChunkRecord]:

        return load_models(
            path,
            ChunkRecord,
        )

class DoublePassMergingChunker(
    BaseSemanticChunker,
):

    def __init__(
        self,
        config: DoublePassMergingChunkerConfig,
        embedder: Embedder,
    ):
        super().__init__(config, embedder)

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
    ) -> list[ChunkRecord]:

        logger.info(
            "Chunking %d nodes using %s...",
            len(nodes),
            self.__class__.__name__,
        )

        records: list[ChunkRecord] = []

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

                chunk_id = generate_chunk_id(node.node_id, idx)

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=node.document_id,
                    node_type=node.node_type,
                    text=llama_node.text,
                    metadata={
                        "page_number": node.page_number,
                        "chunk_index": idx,
                    },
                )

                embedding = ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    embedding=self.embedder.embed_query(
                        chunk.text,
                    ),
                )

                records.append(
                    ChunkRecord(
                        chunk=chunk,
                        embedding=embedding,
                    )
                )

                logger.debug(
                    "Created chunk %s (%d chars).",
                    chunk_id,
                    len(llama_node.text),
                )

        logger.info(
            "Finished semantic chunking: %d nodes -> %d chunks.",
            len(nodes),
            len(records),
        )

        return records