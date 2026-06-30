from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class VectorRecord(BaseModel):
    """
    Generic vector record stored in a vector database.
    """

    id: str

    embedding: list[float]

    payload: dict[str, Any]


class SearchResult(BaseModel):
    """
    Generic search result returned from a vector database.
    """

    id: str

    score: float

    payload: dict[str, Any]

    embedding: list[float] | None = None


class BaseVectorStore(ABC):

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:
        pass

    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        pass

    @abstractmethod
    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        pass

    @abstractmethod
    def list_collections(
        self,
    ) -> list[str]:
        pass

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        batch_size: int = 256,
    ) -> None:
        """
        Insert or update vectors.
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> list[SearchResult]:
        """
        Dense vector search.
        """
        pass

    @abstractmethod
    def scroll(
        self,
        collection_name: str,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> list[SearchResult]:
        """
        Iterate over all vectors satisfying a filter.
        """
        pass

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

class QdrantVectorStore(BaseVectorStore):

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        path: str | None = None,
        timeout: float = 120.0,
    ):
        if path is None:
            self.client = QdrantClient(
                host=host,
                port=port,
                timeout=timeout,
            )
        else:
            self.client = QdrantClient(
                path=path,
            )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:

        if self.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:

        if self.collection_exists(collection_name):
            self.client.delete_collection(
                collection_name,
            )

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:

        collections = {
            c.name
            for c in self.client.get_collections().collections
        }

        return collection_name in collections

    def list_collections(
        self,
    ) -> list[str]:

        return sorted(
            c.name
            for c in self.client.get_collections().collections
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        batch_size: int = 256,
    ) -> None:

        for start in range(
            0,
            len(records),
            batch_size,
        ):

            batch = records[
                start:start + batch_size
            ]

            points = [
                PointStruct(
                    id=record.id,
                    vector=record.embedding,
                    payload=record.payload,
                )
                for record in batch
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> list[SearchResult]:

        query_filter = self._build_filter(
            filters,
        )

        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=with_vectors,
        )

        return [
            SearchResult(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload,
                embedding=(
                    hit.vector
                    if with_vectors
                    else None
                ),
            )
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll(
        self,
        collection_name: str,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> list[SearchResult]:

        query_filter = self._build_filter(
            filters,
        )

        results: list[SearchResult] = []

        offset = None

        while True:

            points, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=with_vectors,
            )

            results.extend(
                SearchResult(
                    id=str(point.id),
                    score=0.0,
                    payload=point.payload,
                    embedding=(
                        point.vector
                        if with_vectors
                        else None
                    ),
                )
                for point in points
            )

            if offset is None:
                break

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(
        filters: dict[str, Any] | None,
    ) -> Filter | None:

        if not filters:
            return None

        conditions = []

        for key, value in filters.items():

            if isinstance(
                value,
                list,
            ):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(
                            any=value,
                        ),
                    )
                )

            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(
                            value=value,
                        ),
                    )
                )

        return Filter(
            must=conditions,
        )