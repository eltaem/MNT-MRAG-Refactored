from abc import ABC, abstractmethod
from typing import Any, Counter, Generic, TypeVar

from qa_parser import QARecord
from src.record import BaseRecord
from src.chunker import ChunkRecord
from src.configs import BaseRetrieverConfig, DenseRetrieverConfig, QARetrieverConfig, SingleDocumentRetrieverConfig, SourceRetrieverConfig
from src.vector_store import BaseVectorStore
from src.nodes import MODALITY_NODE_TYPES

RecordT = TypeVar("RecordT", bound=BaseRecord,)

class BaseRetriever(
    ABC,
    Generic[RecordT],
):

    def __init__(
        self,
        config: BaseRetrieverConfig,
        vector_store: BaseVectorStore,
        collection_name: str,
        record_type: type[RecordT],
    ):
        self.config = config
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.record_type = record_type

    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
    ) -> list[RecordT]:
        ...

    @abstractmethod
    def scroll(
        self,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordT]:
        ...

class DenseRetriever(
    BaseRetriever[RecordT],
    Generic[RecordT],
):

    def __init__(
        self,
        config: DenseRetrieverConfig,
        vector_store: BaseVectorStore,
        collection_name: str,
        record_type: type[RecordT],
    ):
        super().__init__(
            config=config,
            vector_store=vector_store,
            collection_name=collection_name,
            record_type=record_type
        )

    def retrieve(
        self,
        query: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RecordT]:

        hits = self.vector_store.search(
            collection_name=self.collection_name,
            query=query,
            top_k=top_k or self.config.top_k,
            filters=filters,
            with_vectors=True,
        )

        return [
            self.record_type.from_search_result(
                hit,
            )
            for hit in hits
        ]
    
    def scroll(
        self,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordT]:

        results = self.vector_store.scroll(
            collection_name=self.collection_name,
            limit=limit,
            filters=filters,
            with_vectors=True,
        )

        return [
            self.record_type.from_search_result(
                result,
            )
            for result in results
        ]

class SourceRetriever(
    DenseRetriever[ChunkRecord],
):

    def __init__(
        self,
        config: SourceRetrieverConfig,
        vector_store: BaseVectorStore,
        collection_name: str,
    ):
        super().__init__(
            config=config,
            vector_store=vector_store,
            collection_name=collection_name,
            record_type=ChunkRecord,
        )

        self.config = config
    
    def _build_source_filters(
        self,
    ) -> dict[str, Any] | None:

        if self.config.modality is None:
            return None

        return {
            "chunk.node_type": list(
                MODALITY_NODE_TYPES[
                    self.config.modality
                ]
            ),
        }
    
    def _merge_source_filters(
        self,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:

        source_filters = self._build_source_filters()

        if source_filters is None:
            return filters

        return {
            **(filters or {}),
            **source_filters,
        }

    def scroll(
        self,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkRecord]:

        return super().scroll(
            limit=limit,
            filters=self._merge_source_filters(filters),
        )
    
    def retrieve(
        self,
        query: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[ChunkRecord]:

        return super().retrieve(
            query=query,
            filters=self._merge_source_filters(filters),
            top_k=top_k,
        )

class QARetriever(
    DenseRetriever[QARecord],
):

    def __init__(
        self,
        config: QARetrieverConfig,
        vector_store: BaseVectorStore,
        collection_name: str,
    ):
        super().__init__(
            config=config,
            vector_store=vector_store,
            collection_name=collection_name,
            record_type=QARecord,
        )

class SingleDocumentRetriever(
    SourceRetriever,
):

    def __init__(
        self,
        config: SingleDocumentRetrieverConfig,
        vector_store: BaseVectorStore,
        collection_name: str,
    ):
        super().__init__(
            config=config,
            vector_store=vector_store,
            collection_name=collection_name,
        )

        self.config = config

    def retrieve(
        self,
        query: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[ChunkRecord]:

        candidate_chunks = super().retrieve(
            query=query,
            filters=filters,
            top_k=self.config.document_candidate_k,
        )

        if not candidate_chunks:
            return []

        document_counts = Counter(
            chunk.document_id
            for chunk in candidate_chunks
        )

        document_id, _ = document_counts.most_common(1)[0]

        return super().retrieve(
            query=query,
            filters={
                **(filters or {}),
                "chunk.document_id": document_id,
            },
            top_k=top_k or self.config.top_k,
        )