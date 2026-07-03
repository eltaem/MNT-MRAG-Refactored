from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from numpy import mean
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field
)

from src.configs import (
    BaseChunkerConfig,
    BaseEvaluatorConfig,
    BaseRetrieverConfig,
    BaseTextGeneratorConfig,
    EmbedderConfig
)

from src.utility import (
    generate_collection_name,
    load_model,
    save_model
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

class BaseEvaluator(ABC):

    def __init__(
        self,
        config: BaseEvaluatorConfig,
    ):
        self.config = config

    @abstractmethod
    def compute_metrics(
        self,
        *args,
        **kwargs,
    ):
        ...

    @abstractmethod
    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> BaseEvaluationResult:
        ...

class RAGEvaluationSample(BaseModel):

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