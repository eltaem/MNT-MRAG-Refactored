from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from src.embedder import Embedder
from src.vector_store import BaseVectorStore
from src.configs import BaseIndexerConfig

class BaseIndexer(ABC):

    def __init__(
        self,
        config: BaseIndexerConfig,
        embedder: Embedder,
        vector_store: BaseVectorStore,
    ):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store

    @abstractmethod
    def index(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Index data into the vector store.
        """
        ...

    async def aindex(
        self,
        *args,
        **kwargs,
    ) -> None:
        return await asyncio.to_thread(
            self.index,
            *args,
            **kwargs,
        )