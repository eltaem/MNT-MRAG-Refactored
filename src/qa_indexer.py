from __future__ import annotations

from src.indexer import BaseIndexer
from src.qa_parser import QAEmbedding, QAEmbeddingParser, QAItem, QAParser, QARecord
from src.embedder import Embedder
from src.vector_store import BaseVectorStore, QdrantVectorStore
from src.configs import EMBEDDER_CONFIGS, STORAGE_CONFIG, QAIndexerConfig, QdrantConfig, QdrantVectorStoreConfig
from qdrant_client.models import PointStruct

from src.utility import clear_gpu

class QAIndexer(BaseIndexer):

    def __init__(
        self,
        config: QAIndexerConfig,
        embedder: Embedder,
        vector_store: BaseVectorStore,
    ):
        super().__init__(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
        )

        self.config = config
    
    def embed_qa_items(
        self,
        qa_items: list[QAItem],
    ) -> list[QARecord]:

        question_embeddings = self.embedder.embed_documents(
            [qa.question for qa in qa_items]
        )

        answer_embeddings = self.embedder.embed_documents(
            [qa.answer for qa in qa_items]
        )

        return [
            QARecord(
                qa=qa,
                embedding=QAEmbedding(
                    qa_id=qa.qa_id,
                    question_embedding=q_emb,
                    answer_embedding=a_emb,
                ),
            )
            for qa, q_emb, a_emb in zip(
                qa_items,
                question_embeddings,
                answer_embeddings,
            )
        ]

    def index(
        self,
        collection_name: str,
        records: list[QARecord],
    ) -> None:

        if (
            not self.config.enabled
            or not records
        ):
            return

        self.vector_store.create_collection_if_not_exists(
            collection_name=collection_name,
            dimension=self.embedder.embedding_dimension,
        )

        batch_size = self.config.batch_size

        for i in range(
            0,
            len(records),
            batch_size,
        ):

            batch = records[
                i : i + batch_size
            ]

            points = [
                PointStruct(
                    id=record.qa.qa_id,
                    vector=record.embedding.question_embedding,
                    payload={
                        "question": record.qa.question,
                        "answer": record.qa.answer,
                        "answer_embedding": record.embedding.answer_embedding,
                    },
                )
                for record in batch
            ]

            self.vector_store.upsert(
                collection_name=collection_name,
                points=points,
            )

if __name__ == "__main__":
    

    qa_items = QAParser.from_json(
        STORAGE_CONFIG.qa_json
    )

    indexer_config = QAIndexerConfig()

    qdrant_config = QdrantConfig()

    vector_store = QdrantVectorStore(
        QdrantVectorStoreConfig(
            backend=qdrant_config,
        )
    )

    for embedder_config in EMBEDDER_CONFIGS:

        print(f"\n== {embedder_config.model_name} ==")

        embedder = Embedder(embedder_config)

        indexer = QAIndexer(
            config=indexer_config,
            embedder=embedder,
            vector_store=vector_store,
        )

        embedding_path = STORAGE_CONFIG.qa_embedding_path(
            embedder_config.model_name
        )

        if embedding_path.exists():

            print("Loading cached embeddings...")

            embeddings = QAEmbeddingParser.from_json(
                embedding_path
            )

            records = QARecord.from_embeddings(
                qa_items,
                embeddings,
            )

        else:

            print("Generating embeddings...")

            records = indexer.embed_qa_items(
                qa_items,
            )

            embedding_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            QAEmbeddingParser.to_json(
                [r.embedding for r in records],
                embedding_path,
            )

        print("Indexing to vector store...")

        indexer.index(
            collection_name=embedder.collection_name("qa"),
            records=records,
        )

        print("Done.")

        print("Cleaning up...")

        embedder.unload()
        
        del embedder
        del indexer
        del records

        clear_gpu()