from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from src.utility import (
    save_models,
    load_models,
)
from src.vector_store import SearchResult


class BaseRecord(
    BaseModel,
    ABC,
):

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> list[Self]:

        return load_models(
            path,
            cls,
        )

    @staticmethod
    def to_json(
        records: list[Self],
        path: str | Path,
    ) -> Path:

        return save_models(
            records,
            path,
        )

    @classmethod
    @abstractmethod
    def from_embeddings(
        cls,
        *args,
        **kwargs,
    ) -> list[Self]:
        ...

    @classmethod
    @abstractmethod
    def from_search_result(
        cls,
        result: SearchResult,
    ) -> Self:
        ...

    @abstractmethod
    def to_payload(
        self,
    ) -> dict:
        ...