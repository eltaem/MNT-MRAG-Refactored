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

        retrieved_records = self.retrieve_chunks(
            qa_record,
        )

        retrieved_chunks = [
            record.text
            for record in retrieved_records
        ]

        prompt = self.build_prompt(
            question=qa_record.qa.question,
            chunks=retrieved_chunks,
        )

        generated_answer = self.generate_answer(
            prompt,
        )

        return self.build_sample(
            qa_record=qa_record,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
        )

    def generate_samples(
        self,
        limit: int | None = None,
    ) -> list[RAGEvaluationSample]:

        qa_records = self.qa_retriever.scroll(
            limit=limit,
        )

        return [
            self.generate_sample(
                qa_record,
            )
            for qa_record in qa_records
        ]