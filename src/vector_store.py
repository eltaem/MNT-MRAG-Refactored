from __future__ import annotations

from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        embeddings,
        metadata=None,
    ) -> None:
        pass

    async def aadd(
        self,
        embeddings,
        metadata=None,
    ) -> None:
        self.add(
            embeddings,
            metadata,
        )