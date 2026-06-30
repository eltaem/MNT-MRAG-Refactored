from abc import ABC, abstractmethod

from nodes import BaseNode


class BaseIndexer(ABC):

    @abstractmethod
    def index(
        self,
        nodes: list[BaseNode],
    ) -> None:
        ...