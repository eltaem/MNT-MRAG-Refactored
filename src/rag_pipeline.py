from __future__ import annotations

from src.configs import RAGPipelineConfig
from src.evaluator import RAGEvaluationSample
from qa_parser import QARecord
from src.retriever import (
    ChunkRecord,
    QARetriever,
    SourceRetriever,
)
from src.models import BaseTextGeneratorModel
from src.utility import logger

class RAGPipeline:

    def __init__(
        self,
        config: RAGPipelineConfig,
        qa_retriever: QARetriever,
        source_retriever: SourceRetriever,
        generator: BaseTextGeneratorModel,
    ):
        self.config = config

        self.qa_retriever = qa_retriever
        self.source_retriever = source_retriever

        self.generator = generator

    # =====================================================
    # Retrieval
    # =====================================================

    def retrieve_chunks(
        self,
        qa_record: QARecord,
    ) -> list[ChunkRecord]:

        return self.source_retriever.retrieve(
            qa_record.embedding.question_embedding,
        )

    # =====================================================
    # Prompt Construction
    # =====================================================

    def build_prompt(
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

    def generate_answer(
        self,
        prompt: str,
    ) -> str:

        return self.generator.generate(
            prompt=prompt,
        )

    # =====================================================
    # Packaging
    # =====================================================

    def build_sample(
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
    ) -> RAGEvaluationSample:

        logger.info(
            "Generating sample for QA '%s'.",
            qa_record.qa.qa_id,
        )

        logger.debug(
            "Question: %s",
            qa_record.qa.question,
        )

        retrieved_records = self.retrieve_chunks(
            qa_record,
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

        prompt = self.build_prompt(
            question=qa_record.qa.question,
            chunks=retrieved_chunks,
        )

        logger.debug(
            "Prompt constructed (%d characters).",
            len(prompt),
        )

        generated_answer = self.generate_answer(
            prompt,
        )

        logger.debug(
            "Generated answer (%d characters).",
            len(generated_answer),
        )

        sample = self.build_sample(
            qa_record=qa_record,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
        )

        logger.info(
            "Finished sample for QA '%s'.",
            qa_record.qa.qa_id,
        )

        return sample


    def generate_samples(
        self,
        limit: int | None = None,
    ) -> list[RAGEvaluationSample]:

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

        samples: list[RAGEvaluationSample] = []

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

            samples.append(
                self.generate_sample(
                    qa_record,
                )
            )

        logger.info(
            "Generated %d evaluation sample(s).",
            len(samples),
        )

        return samples