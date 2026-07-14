from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Any, Generic, Self, TypeVar
import sacrebleu
from rouge_score import rouge_scorer
from src.utility import logger

from numpy import mean
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field
)

from src.ragas_adapter import RAGASLLMAdapter
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    SemanticSimilarity
)

from src.models import BaseTextGeneratorModel
from src.embedder import Embedder
from src.configs import (
    BaseChunkerConfig,
    BaseEvaluatorConfig,
    BaseRetrieverConfig,
    BaseTextGeneratorConfig,
    EmbedderConfig,
    RAGEvaluatorConfig
)

from src.utility import (
    generate_collection_name,
    load_model,
    load_models,
    save_model,
    save_models
)

class BaseEvaluationResult(BaseModel, ABC):

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @abstractmethod
    def to_markdown(self) -> str:
        ...

    def to_json(
        self,
        path: str | Path,
    ) -> Path:

        return save_model(
            self,
            path,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ):
        return load_model(
            path,
            cls,
        )

    def save(
        self,
        directory: str | Path,
    ) -> Path:

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.to_json(
            directory / "metrics.json"
        )

        (directory / "report.md").write_text(
            self.to_markdown(),
            encoding="utf-8",
        )

        return directory

ResultT = TypeVar(
    "ResultT",
    bound=BaseEvaluationResult,
)

class BaseEvaluator(
    ABC,
    Generic[ResultT],
):

    def __init__(
        self,
        config: BaseEvaluatorConfig,
    ):
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> ResultT:
        ...

    @abstractmethod
    def evaluate_many(
        self,
        *args,
        **kwargs,
    ) -> list[ResultT]:
        ...


class BaseSample(
    BaseModel,
    ABC,
):

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> list[Self]:

        return load_models(
            path,
            cls,
        )

    @staticmethod
    def to_json(
        samples: list[Self],
        path: str | Path,
    ) -> Path:

        return save_models(
            samples,
            path,
        )


class RAGEvaluationSample(BaseSample):

    qa_id: str

    question: str

    ground_truth: str

    retrieved_chunks: list[str]

    generated_answer: str

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

class RAGMetrics(BaseModel):

    # =====================================================
    # Lexical Metrics
    # =====================================================

    bleu: float

    rouge1: float

    rouge2: float

    rougeL: float

    # =====================================================
    # RAGAS Metrics
    # =====================================================

    context_precision: float

    context_recall: float

    faithfulness: float

    answer_relevancy: float

    semantic_similarity: float

class RAGEvaluationResult(BaseEvaluationResult):

    sample: RAGEvaluationSample

    metrics: RAGMetrics

class RAGEvaluationSetup(BaseModel):

    embedder: EmbedderConfig

    chunker: BaseChunkerConfig

    generator: BaseTextGeneratorConfig

    source_retriever: BaseRetrieverConfig

    qa_retriever: BaseRetrieverConfig

    @computed_field
    @property
    def source_collection(self) -> str:
        return generate_collection_name(
            "src",
            self.chunker,
            self.embedder,
        )

    @computed_field
    @property
    def qa_collection(self) -> str:
        return generate_collection_name(
            "qa",
            self.embedder,
        )

class RAGEvaluationSummary(BaseModel):

    sample_count: int

    average_metrics: RAGMetrics

class RAGEvaluationRun(BaseModel):

    setup: RAGEvaluationSetup

    results: list[RAGEvaluationResult]

    @computed_field
    @property
    def summary(self) -> RAGEvaluationSummary:

        metrics = [r.metrics for r in self.results]

        return RAGEvaluationSummary(
            sample_count=len(metrics),
            average_metrics=RAGMetrics(
                bleu=mean(m.bleu for m in metrics),
                rouge1=mean(m.rouge1 for m in metrics),
                rouge2=mean(m.rouge2 for m in metrics),
                rougeL=mean(m.rougeL for m in metrics),
                context_precision=mean(m.context_precision for m in metrics),
                context_recall=mean(m.context_recall for m in metrics),
                faithfulness=mean(m.faithfulness for m in metrics),
                answer_relevancy=mean(m.answer_relevancy for m in metrics),
                semantic_similarity=mean(m.semantic_similarity for m in metrics),
            ),
        )

class BenchmarkResult(BaseModel):

    runs: list[RAGEvaluationRun]

class RAGEvaluator(
    BaseEvaluator[RAGEvaluationResult],
):
    def __init__(
        self,
        config: RAGEvaluatorConfig,
        embedder: Embedder,
        generator: BaseTextGeneratorModel,
    ):
        super().__init__(config)

        self.config = config

        self.llm = RAGASLLMAdapter(
            generator=generator,
        )

        self.embedder = embedder

        self.faithfulness = Faithfulness(
            llm=self.llm,
        )

        self.answer_relevancy = AnswerRelevancy(
            llm=self.llm,
            embeddings=self.embedder,
        )

        self.semantic_similarity = SemanticSimilarity(
            embeddings=self.embedder,
        )

        self.context_precision = ContextPrecision(
            llm=self.llm,
        )

        self.context_recall = ContextRecall(
            llm=self.llm,
        )

        self.rouge = rouge_scorer.RougeScorer(
            [
                "rouge1",
                "rouge2",
                "rougeL",
            ],
            use_stemmer=True,
        )

    # =====================================================
    # RAGAS Metrics
    # =====================================================

    async def _compute_ragas_metrics(
        self,
        sample: RAGEvaluationSample,
    ) -> dict[str, float]:

        (
            faithfulness,
            answer_relevancy,
            semantic_similarity,
            context_precision,
            context_recall,
        ) = await asyncio.gather(

            self.faithfulness.ascore(
                user_input=sample.question,
                response=sample.generated_answer,
                retrieved_contexts=sample.retrieved_chunks,
            ),

            self.answer_relevancy.ascore(
                user_input=sample.question,
                response=sample.generated_answer,
            ),

            self.semantic_similarity.ascore(
                reference=sample.ground_truth,
                response=sample.generated_answer,
            ),

            self.context_precision.ascore(
                user_input=sample.question,
                reference=sample.ground_truth,
                retrieved_contexts=sample.retrieved_chunks,
            ),

            self.context_recall.ascore(
                user_input=sample.question,
                retrieved_contexts=sample.retrieved_chunks,
                reference=sample.ground_truth,
            ),
        )

        return {
            "context_precision": context_precision.value,
            "context_recall": context_recall.value,
            "faithfulness": faithfulness.value,
            "answer_relevancy": answer_relevancy.value,
            "semantic_similarity": semantic_similarity.value,
        }

    # =====================================================
    # Classical Metrics
    # =====================================================

    def _compute_bleu_rouge(
        self,
        sample: RAGEvaluationSample,
    ) -> dict[str, float]:

        bleu = sacrebleu.sentence_bleu(
            sample.generated_answer,
            [sample.ground_truth],
        )

        rouge = self.rouge.score(
            sample.ground_truth,
            sample.generated_answer,
        )

        return {
            "bleu": bleu.score,

            "rouge1":
                rouge["rouge1"].fmeasure,

            "rouge2":
                rouge["rouge2"].fmeasure,

            "rougeL":
                rouge["rougeL"].fmeasure,
        }

    # =====================================================
    # Packaging
    # =====================================================

    def _build_metrics(
        self,
        ragas_metrics: dict[str, float],
        classical_metrics: dict[str, float],
    ) -> RAGMetrics:

        return RAGMetrics(

            bleu=classical_metrics["bleu"],

            rouge1=classical_metrics["rouge1"],

            rouge2=classical_metrics["rouge2"],

            rougeL=classical_metrics["rougeL"],

            context_precision=ragas_metrics["context_precision"],

            context_recall=ragas_metrics["context_recall"],

            faithfulness=ragas_metrics["faithfulness"],

            answer_relevancy=ragas_metrics["answer_relevancy"],

            semantic_similarity=ragas_metrics["semantic_similarity"],
        )

    def _build_result(
        self,
        sample: RAGEvaluationSample,
        metrics: RAGMetrics,
    ) -> RAGEvaluationResult:

        return RAGEvaluationResult(
            sample=sample,
            metrics=metrics,
        )

    # =====================================================
    # Public API
    # =====================================================

    def evaluate(
        self,
        sample: RAGEvaluationSample,
    ) -> RAGEvaluationResult:

        logger.info(
            "Evaluating QA '%s'.",
            sample.qa_id,
        )

        logger.debug(
            "Question: %s",
            sample.question,
        )

        logger.debug(
            "Running RAGAS metrics.",
        )

        ragas_metrics = asyncio.run(
            self._compute_ragas_metrics(
                sample,
            )
        )

        logger.debug(
            "RAGAS metrics: %s",
            ragas_metrics,
        )

        logger.debug(
            "Running BLEU and ROUGE.",
        )

        classical_metrics = self._compute_bleu_rouge(
            sample,
        )

        logger.debug(
            "Classical metrics: %s",
            classical_metrics,
        )

        metrics = self._build_metrics(
            ragas_metrics,
            classical_metrics,
        )

        result = self._build_result(
            sample,
            metrics,
        )

        logger.info(
            "Finished evaluation for QA '%s'.",
            sample.qa_id,
        )

        return result

    def evaluate_many(
        self,
        samples: list[RAGEvaluationSample],
    ) -> list[RAGEvaluationResult]:

        logger.info(
            "Evaluating %d sample(s).",
            len(samples),
        )

        results: list[RAGEvaluationResult] = []

        total = len(samples)

        for index, sample in enumerate(
            samples,
            start=1,
        ):

            logger.info(
                "[%d/%d] Evaluating QA '%s'.",
                index,
                total,
                sample.qa_id,
            )

            results.append(
                self.evaluate(
                    sample,
                )
            )

        logger.info(
            "Finished evaluating %d sample(s).",
            len(results),
        )

        return results