from src.chunker import BaseSemanticChunker
from src.configs import (
    EMBEDDER_CONFIGS,
    DoublePassMergingChunkerConfig,
    QdrantConfig,
    QdrantVectorStoreConfig,
    STORAGE_CONFIG,
    SourceIndexerConfig,
)
from src.embedder import Embedder
from src.utility import (
    clear_gpu,
    generate_collection_name,
    logger,
)
from src.vector_store import BaseVectorStore, QdrantVectorStore
from qdrant_client.models import PointStruct

from src.chunker import ChunkRecord
from src.configs import SourceIndexerConfig
from src.embedder import Embedder
from src.indexer import BaseIndexer

class SourceIndexer(BaseIndexer):

    def __init__(
        self,
        config: SourceIndexerConfig,
        embedder: Embedder,
        vector_store: BaseVectorStore,
    ):
        super().__init__(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
        )

        self.config = config

    def index(
        self,
        collection_name: str,
        records: list[ChunkRecord],
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
                    id=record.chunk_id,
                    vector=record.embedding.embedding,
                    payload=record.to_payload(),
                )
                for record in batch
            ]

            self.vector_store.upsert(
                collection_name=collection_name,
                points=points,
            )

def index_sources():

    STORAGE_CONFIG.ensure_directories()

    logger.info(
        "Starting source indexing..."
    )

    indexer_config = SourceIndexerConfig()

    vector_store = QdrantVectorStore(
        QdrantVectorStoreConfig(
            backend=QdrantConfig(),
        )
    )

    chunker_config = DoublePassMergingChunkerConfig()

    for embedder_config in EMBEDDER_CONFIGS:

        logger.info(
            "=================================================="
        )

        logger.info(
            "Embedder: %s",
            embedder_config.model_name,
        )

        embedder = Embedder(
            embedder_config,
        )

        indexer = SourceIndexer(
            config=indexer_config,
            embedder=embedder,
            vector_store=vector_store,
        )

        chunk_dir = STORAGE_CONFIG.chunk_embedder_dir(
            method_name=chunker_config.name,
            embedder_name=embedder.name,
        )

        chunk_files = sorted(
            chunk_dir.glob("*.json")
        )

        if not chunk_files:

            logger.warning(
                "No chunk records found in %s",
                chunk_dir,
            )

            embedder.unload()

            del embedder
            del indexer

            clear_gpu()

            continue

        collection_name = generate_collection_name(
            "src",
            chunker_config,
            embedder,
        )

        logger.info(
            "Collection: %s",
            collection_name,
        )

        for chunk_path in chunk_files:

            logger.info(
                "Loading %s",
                chunk_path.name,
            )

            records = BaseSemanticChunker.from_json(
                chunk_path,
            )

            logger.info(
                "Indexing %d chunk records...",
                len(records),
            )

            indexer.index(
                collection_name=collection_name,
                records=records,
            )

        logger.info(
            "Finished indexing collection '%s'.",
            collection_name,
        )

        logger.info(
            "Cleaning up embedder..."
        )

        embedder.unload()

        del embedder
        del indexer

        clear_gpu()

    logger.info(
        "Source indexing completed."
    )


if __name__ == "__main__":
    index_sources()