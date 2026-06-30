from __future__ import annotations

from abc import ABC, abstractmethod

from src.nodes import BaseNode

class BaseChunkBuilder(ABC):
    @abstractmethod
    def build(
        self,
        nodes: list[BaseNode],
    ):
        pass

    async def abuild(
        self,
        nodes: list[BaseNode],
    ):
        return self.build(
            nodes
        )