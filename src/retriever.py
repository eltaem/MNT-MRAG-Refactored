from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from qa_parser import QARecord
from src.chunker import ChunkRecord
from src.configs import BaseRetrieverConfig, DenseRetrieverConfig, QARetrieverConfig, SourceRetrieverConfig
from src.vector_store import BaseVectorStore

RecordT = TypeVar("RecordT")

class BaseRetriever(
    ABC,
    Generic[RecordT],
):

    def __init__(
        self,
        config: BaseRetrieverConfig,
        vector_store: BaseVectorStore,
        collection_name: str,
        record_cls: type[RecordT],
    ):
        self.config = config
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.record_cls = record_cls

    @abstractmethod
    def retrieve(
        self,
        query_vector: list[float],
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
        record_cls: type[RecordT],
    ):
        super().__init__(
            config=config,
            vector_store=vector_store,
            collection_name=collection_name,
            record_cls=record_cls
        )

        self.config = config
        self.collection_name = collection_name
        self.record_cls = record_cls

    def retrieve(
        self,
        query_vector: list[float],
    ) -> list[RecordT]:

        hits = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.config.top_k,
            score_threshold=self.config.score_threshold,
        )

        return [
            self.record_cls.from_search_result(
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
            self.record_cls.from_search_result(
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
            record_cls=ChunkRecord,
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
            record_cls=QARecord,
        )