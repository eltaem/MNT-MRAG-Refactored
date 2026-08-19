from __future__ import annotations

import time
from src.embedder import Embedder
from src.configs import RAGPipelineConfig
from src.rag_schemas import (
    RAGEvaluationSample,
    RAGEvaluationSetup,
    RAGPipelineResult,
)
from qa_parser import QARecord
from src.retriever import (
    ChunkRecord,
    QARetriever,
    SourceRetriever,
)
from src.models import BaseTextGeneratorModel, OllamaTextGenerator
from src.utility import logger, clear_gpu
from src.vector_store import QdrantVectorStore

class RAGPipeline:

    def __init__(
        self,
        config: RAGPipelineConfig,
        embedder: Embedder,
        qa_retriever: QARetriever,
        source_retriever: SourceRetriever,
        generator: BaseTextGeneratorModel,
    ):
        self.config = config
        self.embedder = embedder
        self.qa_retriever = qa_retriever
        self.source_retriever = source_retriever
        self.generator = generator

    # =====================================================
    # Build
    # =====================================================
    @classmethod
    def build(
        cls,
        setup: RAGEvaluationSetup,
    ) -> "RAGPipeline":

        embedder = Embedder(
            config=setup.embedder,
        )

        vector_store = QdrantVectorStore(setup.vector_store)

        source_retriever = SourceRetriever(
            config=setup.source_retriever,
            vector_store=vector_store,
            collection_name=setup.source_collection,
        )

        qa_retriever = QARetriever(
            config=setup.qa_retriever,
            vector_store=vector_store,
            collection_name=setup.qa_collection,
        )

        generator = OllamaTextGenerator(
            setup.answer_generator,
        )

        generator.load_pretrained()

        return cls(
            config=setup.rag,
            embedder=embedder,
            qa_retriever=qa_retriever,
            source_retriever=source_retriever,
            generator=generator,
        )

    # =====================================================
    # Retrieval
    # =====================================================

    def _retrieve_chunks(
        self,
        qa_record: QARecord,
    ) -> list[ChunkRecord]:

        return self.source_retriever.retrieve(
            qa_record.embedding.question_embedding,
        )

    # =====================================================
    # Prompt Construction
    # =====================================================

    def _build_prompt(
        self,
        question: str,
        chunks: list[str],
    ) -> str:

        context = self.config.separator.join(
            chunks,
        )

        return (
            f"{self.config.context_header}:\n"
            f"{context}"
            f"{self.config.separator}"
            f"{self.config.question_header}:\n"
            f"{question}"
            f"{self.config.separator}"
            f"{self.config.answer_header}:"
        )

    # =====================================================
    # Generation
    # =====================================================

    def _generate_answer(
        self,
        prompt: str,
    ) -> str:

        return self.generator.generate(
            prompt=prompt,
        )

    # =====================================================
    # Packaging
    # =====================================================

    def _build_sample(
        self,
        qa_record: QARecord,
        retrieved_chunks: list[str],
        generated_answer: str,
    ) -> RAGEvaluationSample:

        return RAGEvaluationSample(
            qa_id=qa_record.qa.qa_id,
            question=qa_record.qa.question,
            ground_truth=qa_record.qa.answer,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
        )

    # =====================================================
    # Pipeline
    # =====================================================

    def generate_sample(
        self,
        qa_record: QARecord,
    ) -> RAGPipelineResult:

        logger.info(
            "Generating sample for QA '%s'.",
            qa_record.qa.qa_id,
        )

        logger.debug(
            "Question: %s",
            qa_record.qa.question,
        )

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieved_records = self._retrieve_chunks(
            qa_record,
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        logger.debug(
            "Retrieval completed for QA '%s' in %.4f seconds.",
            qa_record.qa.qa_id,
            retrieval_time,
        )

        if not retrieved_records:
            logger.warning(
                "No chunks retrieved for QA '%s'.",
                qa_record.qa.qa_id,
            )
        else:
            logger.debug(
                "Retrieved %d chunk(s) for QA '%s'.",
                len(retrieved_records),
                qa_record.qa.qa_id,
            )

        retrieved_chunks = [
            record.text
            for record in retrieved_records
        ]

        # -------------------------------------------------
        # Prompt construction
        # -------------------------------------------------

        prompt = self._build_prompt(
            question=qa_record.qa.question,
            chunks=retrieved_chunks,
        )

        logger.debug(
            "Prompt constructed (%d characters).",
            len(prompt),
        )

        # -------------------------------------------------
        # Generation
        # -------------------------------------------------

        generation_start = time.perf_counter()

        generated_answer = self._generate_answer(
            prompt,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        logger.debug(
            "Generation completed for QA '%s' in %.4f seconds.",
            qa_record.qa.qa_id,
            generation_time,
        )

        logger.debug(
            "Generated answer (%d characters).",
            len(generated_answer),
        )

        # -------------------------------------------------
        # Sample
        # -------------------------------------------------

        sample = self._build_sample(
            qa_record=qa_record,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
        )

        logger.info(
            "Finished sample for QA '%s'.",
            qa_record.qa.qa_id,
        )

        return RAGPipelineResult(
            sample=sample,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
        )

    def generate_samples(
        self,
        limit: int | None = None,
    ) -> list[RAGPipelineResult]:

        logger.info(
            "Loading QA records%s.",
            (
                f" (limit={limit})"
                if limit is not None
                else ""
            ),
        )

        qa_records = self.qa_retriever.scroll(
            limit=limit,
        )

        logger.info(
            "Loaded %d QA record(s).",
            len(qa_records),
        )

        results: list[RAGPipelineResult] = []

        total = len(qa_records)

        for index, qa_record in enumerate(
            qa_records,
            start=1,
        ):

            logger.info(
                "[%d/%d] Processing QA '%s'.",
                index,
                total,
                qa_record.qa.qa_id,
            )

            results.append(
                self.generate_sample(
                    qa_record,
                )
            )

        logger.info(
            "Generated %d evaluation sample(s).",
            len(results),
        )

        clear_gpu()

        return results