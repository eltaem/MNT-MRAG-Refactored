from __future__ import annotations
import asyncio
from pathlib import Path

from src.processors import BaseNodeProcessor
from src.enrichers import BaseNodeEnricher
from src.extractors import PDFExtractor
from src.nodes import NODE_REGISTRY, BaseNode
from src.configs import PDFProcessorConfig
from src.utility import collect_pdfs, load_json, save_models
import asyncio

from src.utility import logger

from src.configs import (
    STORAGE_CONFIG,
    PDFExtractorConfig,
    PDFProcessorConfig,
    RemoveBeforePatternProcessorConfig,
    TextMergerConfig,
    DEFAULT_FRONT_MATTER_PATTERNS,
    TextMergeMode,
    OllamaImageCaptioningConfig,
    OllamaTableCaptioningConfig,
    ImageCaptionEnricherConfig,
    TableCaptionEnricherConfig,
)

from src.extractors import PDFExtractor

from src.processors import (
    RemoveBeforePatternProcessor,
    TextMerger,
)

from src.enrichers import (
    ImageCaptionEnricher,
    TableCaptionEnricher,
)

from src.models import (
    OllamaImageCaptioningModel,
    OllamaTableCaptioningModel,
)

class PDFProcessor:
    """
    High-level PDF processing orchestrator.

        PDF
         ↓
        PDFExtractor
         ↓
        list[BaseNode]
         ↓
        Processors
         ↓
        Enrichers
    """

    def __init__(
        self,
        config:PDFProcessorConfig,
        extractor: PDFExtractor,
        processors: list[BaseNodeProcessor] | None = None,
        enrichers: list[BaseNodeEnricher] | None = None,
        ):
        
        self.config = config
        self.extractor = extractor

        self.processors = (
            processors
            if processors is not None
            else []
        )

        self.enrichers = (
            enrichers
            if enrichers is not None
            else []
        )

    def process_pdf(self, path: str | Path,) -> list[BaseNode]:

        nodes = self.extractor.extract(path)

        for processor in self.processors:
            nodes = processor.process(nodes)

        for enricher in self.enrichers:
            nodes = enricher.enrich(nodes)

        return nodes

    def process_pdfs(
        self,
        paths: list[str | Path],
    ) -> dict[str, list[BaseNode]]:

        results: dict[str, list[BaseNode]] = {}

        for path in paths:
            results[Path(path).stem] = self.process_pdf(path)

        return results
    
    def process_pdfs_dir(
        self,
        directory: str | Path,
    ) -> dict[str, list[BaseNode]]:

        pdf_paths = collect_pdfs(directory)

        return self.process_pdfs(pdf_paths)

    async def aprocess_pdf(self, path: str | Path,) -> list[BaseNode]:

        nodes = self.extractor.extract(path)

        for processor in (
            self.processors
        ):
            nodes = await processor.aprocess(nodes)

        for enricher in self.enrichers:
            nodes = await enricher.aenrich(nodes)

        return nodes

    async def aprocess_pdfs(
        self,
        paths: list[str | Path],
    ) -> dict[str, list[BaseNode]]:

        if not paths:
            return {}

        semaphore = asyncio.Semaphore(
            self.config.pdf_concurrency
        )

        async def run_one(
            path: str | Path,
        ) -> tuple[str, list[BaseNode]]:

            async with semaphore:

                return (
                    Path(path).stem,
                    await self.aprocess_pdf(path),
                )

        tasks = [
            run_one(path)
            for path in paths
        ]

        results = await asyncio.gather(*tasks)

        return dict(results)
    
    async def aprocess_pdfs_dir(
        self,
        directory: str | Path,
    ) -> dict[str, list[BaseNode]]:

        pdf_paths = collect_pdfs(directory)

        return await self.aprocess_pdfs(pdf_paths)

    @staticmethod
    def to_json(
        nodes: list[BaseNode],
        path: str | Path,
    ) -> Path:

        return save_models(
            nodes,
            path,
        )
    
    @staticmethod
    def from_json(
        path: str | Path,
    ) -> list[BaseNode]:

        payload = load_json(path)

        nodes: list[BaseNode] = []

        for item in payload:

            node_type = item.pop("node_type")
            item.pop("node_id", None)

            node_cls = NODE_REGISTRY[node_type]

            node = node_cls.model_validate(item)
            nodes.append(node)

        return nodes

async def process_pdfs():

    STORAGE_CONFIG.ensure_directories()

    logger.info(
        "Initializing PDF processor..."
    )

    # ======================================================
    # Extractor
    # ======================================================

    extractor = PDFExtractor(
        PDFExtractorConfig()
    )

    # ======================================================
    # Captioning Models
    # ======================================================

    image_captioning_model = OllamaImageCaptioningModel(
        OllamaImageCaptioningConfig(
            model_name="qwen3.5:9b",
            device="cuda",
            max_new_tokens=512,
        )
    )
    image_captioning_model.load_pretrained()

    table_captioning_model = OllamaTableCaptioningModel(
        OllamaTableCaptioningConfig(
            model_name="qwen3.5:9b",
            device="cuda",
            max_new_tokens=512,
        )
    )
    table_captioning_model.load_pretrained()

    # ======================================================
    # Processor
    # ======================================================

    processor = PDFProcessor(
        config=PDFProcessorConfig(),
        extractor=extractor,
        processors=[
            RemoveBeforePatternProcessor(
                RemoveBeforePatternProcessorConfig(
                    patterns=DEFAULT_FRONT_MATTER_PATTERNS,
                ),
            ),
            TextMerger(
                TextMergerConfig(
                    mode=TextMergeMode.DOCUMENT,
                ),
            ),
        ],
        enrichers=[
            ImageCaptionEnricher(
                ImageCaptionEnricherConfig(
                    batch_size=2,
                ),
                image_captioning_model,
            ),
            TableCaptionEnricher(
                TableCaptionEnricherConfig(
                    batch_size=2,
                ),
                table_captioning_model,
            ),
        ],
    )

    # ======================================================
    # Process PDFs
    # ======================================================

    pdf_paths = collect_pdfs(
        STORAGE_CONFIG.source_dir,
    )

    logger.info(
        "Found %d PDF(s).",
        len(pdf_paths),
    )

    for pdf_path in pdf_paths:

        document_name = pdf_path.stem

        output_path = STORAGE_CONFIG.parsed_document_path(
            document_name,
        )

        if output_path.exists():

            logger.info(
                "[SKIP] %s already parsed.",
                document_name,
            )

            continue

        logger.info(
            "Processing %s...",
            pdf_path.name,
        )

        nodes = await processor.aprocess_pdf(
            pdf_path,
        )

        PDFProcessor.to_json(
            nodes,
            output_path,
        )

        logger.info(
            "Saved %s (%d nodes).",
            output_path.name,
            len(nodes),
        )

    # ======================================================
    # Cleanup
    # ======================================================

    image_captioning_model.unload()
    table_captioning_model.unload()

    logger.info(
        "PDF processing completed."
    )


if __name__ == "__main__":
    asyncio.run(process_pdfs())