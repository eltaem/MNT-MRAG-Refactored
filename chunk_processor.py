import asyncio

from src.chunker import BaseSemanticChunker, DoublePassMergingChunker
from src.embedder import Embedder
from pdf_processor import PDFProcessor
from src.configs import EMBEDDER_CONFIGS, STORAGE_CONFIG, DoublePassMergingChunkerConfig
from src.utility import clear_gpu, logger

async def process_chunks():

    STORAGE_CONFIG.ensure_directories()

    logger.info(
        "Starting semantic chunk generation..."
    )

    parsed_files = sorted(
        STORAGE_CONFIG.parsed_dir.glob("*.json")
    )

    if not parsed_files:
        logger.warning(
            "No parsed documents found in %s.",
            STORAGE_CONFIG.parsed_dir,
        )
        return

    chunker_config = DoublePassMergingChunkerConfig()

    for embedder_config in EMBEDDER_CONFIGS:

        logger.info(
            "=================================================="
        )
        logger.info(
            "Embedder: %s",
            embedder_config.model_name,
        )

        embedder = Embedder(embedder_config)

        chunker = DoublePassMergingChunker(
            config=chunker_config,
            embedder=embedder,
        )

        for parsed_path in parsed_files:

            document_name = parsed_path.stem

            output_path = STORAGE_CONFIG.chunk_path(
                method_name=chunker.name,
                document_name=document_name,
                embedder_name=embedder.name,
            )

            if output_path.exists():

                logger.info(
                    "Skipping %s (%s): cached.",
                    document_name,
                    embedder.name,
                )
                continue

            logger.info(
                "Loading parsed document: %s",
                parsed_path.name,
            )

            nodes = PDFProcessor.from_json(
                parsed_path,
            )

            logger.info(
                "Chunking %s (%d nodes)...",
                document_name,
                len(nodes),
            )

            records = chunker.chunk(
                nodes,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            BaseSemanticChunker.to_json(
                records,
                output_path,
            )

            logger.info(
                "Saved %d chunk records -> %s",
                len(records),
                output_path,
            )

        logger.info(
            "Cleaning up embedder..."
        )

        embedder.unload()

        del chunker
        del embedder

        clear_gpu()

    logger.info(
        "Semantic chunk generation completed."
    )

if __name__ == "__main__":
    asyncio.run(process_chunks())