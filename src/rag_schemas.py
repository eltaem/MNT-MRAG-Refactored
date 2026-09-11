from __future__ import annotations
from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Self

from statistics import mean
from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from src.configs import (
    DoublePassMergingChunkerConfig,
    EmbedderConfig,
    OllamaTextGeneratorConfig,
    QARetrieverConfig,
    QdrantVectorStoreConfig,
    RAGPipelineConfig,
    SingleDocumentRetrieverConfig,
)
from src.utility import escape_md, generate_collection_name, load_model, load_models, save_model, save_models

class BaseArtifact(
    BaseModel,
    ABC,
):

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @abstractmethod
    def to_markdown(self) -> str:
        """
        Render this artifact as Markdown.
        """
        ...

    def to_json(
        self,
        path: str | Path,
    ) -> None:

        save_model(
            self,
            path,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> Self:

        return load_model(
            cls,
            path,
        )

    def save_markdown(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            self.to_markdown(),
            encoding="utf-8",
        )

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
            cls,
            path,
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

class RAGMetrics(BaseModel):

    # =====================================================
    # Lexical Metrics
    # =====================================================

    bleu: float = 0.0

    rouge1: float = 0.0

    rouge2: float = 0.0

    rougeL: float = 0.0

    # =====================================================
    # RAGAS Metrics
    # =====================================================

    context_precision: float = 0.0

    context_recall: float = 0.0

    faithfulness: float = 0.0

    answer_relevancy: float = 0.0

    semantic_similarity: float = 0.0

    @field_validator("*")
    @classmethod
    def replace_non_finite(cls, value: float) -> float:
        return value if math.isfinite(value) else 0.0

class RAGRuntime(BaseModel):

    retrieval_time: float

    generation_time: float

    metric_evaluation_time: float

    @computed_field
    @property
    def total_time(self) -> float:
        return (
            self.retrieval_time
            + self.generation_time
            + self.metric_evaluation_time
        )

class RAGPipelineResult(BaseArtifact):

    sample: RAGEvaluationSample
    retrieval_time: float
    generation_time: float

    def to_markdown(self) -> str:
        """Generate markdown report for a single RAG generation result."""

        lines: list[str] = []

        lines.append(
            f"## Question: {self.sample.qa_id}\n"
        )

        lines.append("**User Question:**")
        lines.append(
            f"\n{escape_md(self.sample.question)}\n"
        )

        lines.append("**Generated Answer:**")
        lines.append(
            f"\n{escape_md(self.sample.generated_answer)}\n"
        )

        lines.append("**Ground Truth:**")
        lines.append(
            f"\n{escape_md(self.sample.ground_truth)}\n"
        )

        lines.append(
            f"**Retrieved Chunks "
            f"({len(self.sample.retrieved_chunks)}):**\n"
        )

        for i, chunk in enumerate(
            self.sample.retrieved_chunks,
            start=1,
        ):
            lines.append(
                f"{i}. {escape_md(chunk)}\n"
            )

        lines.append("\n### Runtime\n")

        lines.append(
            "| Retrieval Time | Generation Time | Total Time |"
        )

        lines.append(
            "|:---:|:---:|:---:|"
        )

        total_time = (
            self.retrieval_time
            + self.generation_time
        )

        lines.append(
            f"| {self.retrieval_time:.4f} s | "
            f"{self.generation_time:.4f} s | "
            f"{total_time:.4f} s |"
        )

        lines.append("\n---\n")

        return "\n".join(lines)

class RAGEvaluationResult(BaseArtifact):

    sample: RAGEvaluationSample
    metrics: RAGMetrics
    runtime: RAGRuntime

    def to_markdown(self) -> str:
        """Generate markdown report for a single RAG evaluation result."""
        lines: list[str] = []

        lines.append(f"## Question: {self.sample.qa_id}\n")

        lines.append("**User Question:**")
        lines.append(f"\n{escape_md(self.sample.question)}\n")

        lines.append("\n**Generated Answer:**")
        lines.append(f"\n{escape_md(self.sample.generated_answer)}\n")

        lines.append("\n**Ground Truth:**")
        lines.append(f"\n{escape_md(self.sample.ground_truth)}\n")

        lines.append(
            f"**Retrieved Chunks ({len(self.sample.retrieved_chunks)}):**\n"
        )

        for i, chunk in enumerate(self.sample.retrieved_chunks, 1):
            lines.append(f"{i}. {escape_md(chunk)}\n")

        lines.append("")

        # ─── Metrics Table ───
        lines.append("### Evaluation Scores\n")

        lines.append(
            "| BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | "
            "Context Precision | Context Recall | Faithfulness | "
            "Answer Relevancy | Semantic Similarity |"
        )

        lines.append(
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        )

        m = self.metrics

        lines.append(
            f"| {m.bleu:.4f} | "
            f"{m.rouge1:.4f} | "
            f"{m.rouge2:.4f} | "
            f"{m.rougeL:.4f} | "
            f"{m.context_precision:.4f} | "
            f"{m.context_recall:.4f} | "
            f"{m.faithfulness:.4f} | "
            f"{m.answer_relevancy:.4f} | "
            f"{m.semantic_similarity:.4f} |"
        )

        # ─── Runtime Table ───
        lines.append("\n### Runtime\n")

        lines.append(
            "| Retrieval Time | Generation Time | "
            "Evaluation Time | Total Time |"
        )

        lines.append(
            "|:---:|:---:|:---:|:---:|"
        )

        r = self.runtime

        lines.append(
            f"| {r.retrieval_time:.4f} s | "
            f"{r.generation_time:.4f} s | "
            f"{r.metric_evaluation_time:.4f} s | "
            f"{r.total_time:.4f} s |"
        )

        lines.append("\n---\n")

        return "\n".join(lines)

class RAGEvaluationSetup(BaseModel):

    embedder: EmbedderConfig

    chunker: DoublePassMergingChunkerConfig

    vector_store: QdrantVectorStoreConfig

    rag: RAGPipelineConfig

    answer_generator: OllamaTextGeneratorConfig

    evaluation_generator: OllamaTextGeneratorConfig

    source_retriever: SingleDocumentRetrieverConfig

    qa_retriever: QARetrieverConfig

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

    average_runtime: RAGRuntime

class RAGGenerationRun(BaseArtifact):

    setup: RAGEvaluationSetup
    
    results: list[RAGPipelineResult]

    def to_markdown(self) -> str:
        """Generate markdown report for a RAG generation run."""

        lines: list[str] = []

        lines.append("# RAG Generation Report\n")

        # =====================================================
        # Experiment Setup
        # =====================================================

        lines.append("## Generation Setup\n")

        lines.append(
            f"- **Embedder:** "
            f"`{self.setup.embedder.model_name}`"
        )

        lines.append(
            f"- **Modality:** "
            f"`{self.setup.source_retriever.modality.display_name}`"
        )

        lines.append(
            f"- **Chunker:** "
            f"`{self.setup.chunker.name}`"
        )

        lines.append(
            f"- **Answer Generator:** "
            f"`{self.setup.answer_generator.model_name}`"
        )

        lines.append("\n---\n")

        # =====================================================
        # Individual Results
        # =====================================================

        lines.append("## Generated Results\n")

        if not self.results:
            lines.append(
                "No generation results available.\n"
            )
        else:
            for result in self.results:
                lines.append(
                    result.to_markdown()
                )

        return "\n".join(lines)

class RAGEvaluationRun(BaseArtifact):

    setup: RAGEvaluationSetup

    results: list[RAGEvaluationResult]

    @computed_field
    @property
    def summary(self) -> RAGEvaluationSummary:

        metrics = [r.metrics for r in self.results]
        runtimes = [r.runtime for r in self.results]

        if not metrics:
            return RAGEvaluationSummary(
                sample_count=0,
                average_metrics=RAGMetrics(
                    bleu=0.0,
                    rouge1=0.0,
                    rouge2=0.0,
                    rougeL=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    semantic_similarity=0.0,
                ),
                average_runtime=RAGRuntime(
                    retrieval_time=0.0,
                    generation_time=0.0,
                    metric_evaluation_time=0.0,
                ),
            )

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
            average_runtime=RAGRuntime(
                retrieval_time=mean(
                    r.retrieval_time
                    for r in runtimes
                ),
                generation_time=mean(
                    r.generation_time
                    for r in runtimes
                ),
                metric_evaluation_time=mean(
                    r.metric_evaluation_time
                    for r in runtimes
                ),
            ),
        )
    
    def to_markdown(self) -> str:

        lines: list[str] = []

        lines.append("# RAG Evaluation Report\n")

        # =====================================================
        # Experiment Setup
        # =====================================================

        lines.append("## Evaluation Setup\n")

        lines.append(
            f"- **Embedder:** `{self.setup.embedder.model_name}`"
        )
        lines.append(
            f"- **Modality:** `{self.setup.source_retriever.modality.display_name}`"
        )
        lines.append(
            f"- **Chunker:** `{self.setup.chunker.name}`"
        )
        lines.append(
            f"- **Answer Generator:** `{self.setup.answer_generator.model_name}`"
        )
        lines.append(
            f"- **Evaluation Generator:** `{self.setup.evaluation_generator.model_name}`"
        )

        lines.append("\n---\n")

        # =====================================================
        # Individual Results
        # =====================================================

        lines.append("## Individual Results\n")

        if not self.results:
            lines.append("No evaluation results available.\n")
        else:
            for result in self.results:
                lines.append(result.to_markdown())

        # =====================================================
        # Aggregate Metrics
        # =====================================================

        lines.append("## Aggregate Metrics\n")

        metrics = self.summary.average_metrics

        lines.append(
            "| BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | "
            "Context Precision | Context Recall | Faithfulness | "
            "Answer Relevancy | Semantic Similarity |"
        )

        lines.append(
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        )

        lines.append(
            f"| {metrics.bleu:.4f} | "
            f"{metrics.rouge1:.4f} | "
            f"{metrics.rouge2:.4f} | "
            f"{metrics.rougeL:.4f} | "
            f"{metrics.context_precision:.4f} | "
            f"{metrics.context_recall:.4f} | "
            f"{metrics.faithfulness:.4f} | "
            f"{metrics.answer_relevancy:.4f} | "
            f"{metrics.semantic_similarity:.4f} |"
        )

        # =====================================================
        # Aggregate Runtime
        # =====================================================

        avg_runtime = self.summary.average_runtime

        lines.append("\n### Average Runtime per Sample\n")

        lines.append(
            "| Retrieval | Generation | Metric Evaluation | Total |"
        )

        lines.append(
            "|:---:|:---:|:---:|:---:|"
        )

        lines.append(
            f"| {avg_runtime.retrieval_time:.4f}s | "
            f"{avg_runtime.generation_time:.4f}s | "
            f"{avg_runtime.metric_evaluation_time:.4f}s | "
            f"{avg_runtime.total_time:.4f}s |"
        )

        return "\n".join(lines)

class BenchmarkResult(BaseArtifact):

    runs: list[RAGEvaluationRun]

    def to_markdown(self) -> str:

        lines: list[str] = []

        lines.append("# RAG Benchmark Report\n")

        if not self.runs:
            lines.append("No evaluation runs available.\n")
            return "\n".join(lines)

        lines.append(
            f"**Total Experiments:** {len(self.runs)}\n"
        )

        embedders = {
            run.setup.embedder.model_name
            for run in self.runs
        }

        chunkers = {
            run.setup.chunker.name
            for run in self.runs
        }

        answer_generators = {
            run.setup.answer_generator.model_name
            for run in self.runs
        }

        evaluation_generators = {
            run.setup.evaluation_generator.model_name
            for run in self.runs
        }

        constant_embedder = (
            next(iter(embedders))
            if len(embedders) == 1
            else None
        )

        constant_chunker = (
            next(iter(chunkers))
            if len(chunkers) == 1
            else None
        )

        constant_answer_generator = (
            next(iter(answer_generators))
            if len(answer_generators) == 1
            else None
        )

        constant_evaluation_generator = (
            next(iter(evaluation_generators))
            if len(evaluation_generators) == 1
            else None
        )

        # =====================================================
        # Benchmark configuration
        # =====================================================

        fixed_configuration: list[tuple[str, str]] = []

        if constant_embedder is not None:
            fixed_configuration.append(
                ("Embedder", constant_embedder)
            )

        if constant_chunker is not None:
            fixed_configuration.append(
                ("Chunker", constant_chunker)
            )

        if constant_answer_generator is not None:
            fixed_configuration.append(
                ("Answer Generator", constant_answer_generator)
            )

        if constant_evaluation_generator is not None:
            fixed_configuration.append(
                ("Evaluation Generator", constant_evaluation_generator)
            )

        if fixed_configuration:
            lines.append("**Fixed Configuration:**\n")

            for name, value in fixed_configuration:
                lines.append(f"- **{name}:** {value}")

            lines.append("")

        lines.append("## Experiment Comparison\n")

        # =====================================================
        # Determine table columns
        # =====================================================

        columns: list[str] = []

        if constant_embedder is None:
            columns.append("Embedder")

        columns.append("Modality")

        if constant_chunker is None:
            columns.append("Chunker")

        if constant_answer_generator is None:
            columns.append("Answer Generator")

        if constant_evaluation_generator is None:
            columns.append("Evaluation Generator")

        columns.extend([
            "BLEU",
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "Context Precision",
            "Context Recall",
            "Faithfulness",
            "Answer Relevancy",
            "Semantic Similarity",
        ])

        columns.extend([
            "Avg Retrieval Time",
            "Avg Generation Time",
            "Avg Metric Evaluation Time",
            "Avg Total Time",
        ])

        lines.append(
            "| " + " | ".join(columns) + " |"
        )

        # Configuration columns are left-aligned;
        # metric columns are centered.
        metric_columns = {
            "BLEU",
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "Context Precision",
            "Context Recall",
            "Faithfulness",
            "Answer Relevancy",
            "Semantic Similarity",
        }

        runtime_columns = {
            "Avg Retrieval Time",
            "Avg Generation Time",
            "Avg Metric Evaluation Time",
            "Avg Total Time",
        }

        centered_columns = metric_columns | runtime_columns

        alignment = [
            ":---:"
            if column in centered_columns
            else ":---"
            for column in columns
        ]

        lines.append(
            "| " + " | ".join(alignment) + " |"
        )

        # =====================================================
        # Find best score for each metric
        # =====================================================

        summaries = [
            run.summary.average_metrics
            for run in self.runs
        ]

        max_bleu = max(m.bleu for m in summaries)
        max_rouge1 = max(m.rouge1 for m in summaries)
        max_rouge2 = max(m.rouge2 for m in summaries)
        max_rougeL = max(m.rougeL for m in summaries)

        max_context_precision = max(
            m.context_precision
            for m in summaries
        )

        max_context_recall = max(
            m.context_recall
            for m in summaries
        )

        max_faithfulness = max(
            m.faithfulness
            for m in summaries
        )

        max_answer_relevancy = max(
            m.answer_relevancy
            for m in summaries
        )

        max_semantic_similarity = max(
            m.semantic_similarity
            for m in summaries
        )

        min_retrieval_time = min(
            run.summary.average_runtime.retrieval_time
            for run in self.runs
        )

        min_generation_time = min(
            run.summary.average_runtime.generation_time
            for run in self.runs
        )

        min_metric_evaluation_time = min(
            run.summary.average_runtime.metric_evaluation_time
            for run in self.runs
        )

        min_total_time = min(
            run.summary.average_runtime.total_time
            for run in self.runs
        )

        # =====================================================
        # Formatting helper
        # =====================================================

        def format_metric(
            value: float,
            maximum: float,
        ) -> str:

            formatted = f"{value:.4f}"

            if value == maximum:
                return f"**{formatted}**"

            return formatted
        
        def format_runtime(
            value: float,
            minimum: float,
        ) -> str:

            formatted = f"{value:.4f}s"

            if value == minimum:
                return f"**{formatted}**"

            return formatted

        # =====================================================
        # Individual experiment rows
        # =====================================================

        previous_embedder: str | None = None
        previous_chunker: str | None = None
        previous_answer_generator: str | None = None
        previous_evaluation_generator: str | None = None

        for run in self.runs:

            setup = run.setup
            metrics = run.summary.average_metrics
            runtime = run.summary.average_runtime

            embedder = setup.embedder.model_name
            chunker = setup.chunker.name
            answer_generator = setup.answer_generator.model_name
            evaluation_generator = (
                setup.evaluation_generator.model_name
            )

            row: list[str] = []

            # -------------------------------------------------
            # Configuration columns
            # -------------------------------------------------

            if constant_embedder is None:
                display_embedder = (
                    embedder
                    if embedder != previous_embedder
                    else ""
                )

                row.append(display_embedder)

            row.append(
                setup.source_retriever.modality.display_name
            )

            if constant_chunker is None:
                display_chunker = (
                    chunker
                    if chunker != previous_chunker
                    else ""
                )

                row.append(display_chunker)

            if constant_answer_generator is None:
                display_answer_generator = (
                    answer_generator
                    if answer_generator != previous_answer_generator
                    else ""
                )

                row.append(display_answer_generator)

            if constant_evaluation_generator is None:
                display_evaluation_generator = (
                    evaluation_generator
                    if evaluation_generator != previous_evaluation_generator
                    else ""
                )

                row.append(display_evaluation_generator)

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            row.extend([
                format_metric(metrics.bleu, max_bleu),
                format_metric(metrics.rouge1, max_rouge1),
                format_metric(metrics.rouge2, max_rouge2),
                format_metric(metrics.rougeL, max_rougeL),
                format_metric(
                    metrics.context_precision,
                    max_context_precision,
                ),
                format_metric(
                    metrics.context_recall,
                    max_context_recall,
                ),
                format_metric(
                    metrics.faithfulness,
                    max_faithfulness,
                ),
                format_metric(
                    metrics.answer_relevancy,
                    max_answer_relevancy,
                ),
                format_metric(
                    metrics.semantic_similarity,
                    max_semantic_similarity,
                ),
            ])

            row.extend([
                format_runtime(
                    runtime.retrieval_time,
                    min_retrieval_time,
                ),
                format_runtime(
                    runtime.generation_time,
                    min_generation_time,
                ),
                format_runtime(
                    runtime.metric_evaluation_time,
                    min_metric_evaluation_time,
                ),
                format_runtime(
                    runtime.total_time,
                    min_total_time,
                ),
            ])

            lines.append(
                "| " + " | ".join(row) + " |"
            )

            previous_embedder = embedder
            previous_chunker = chunker
            previous_answer_generator = answer_generator
            previous_evaluation_generator = (
                evaluation_generator
            )

        # =====================================================
        # Benchmark Runtime
        # =====================================================

        runtimes = [
            run.summary.average_runtime
            for run in self.runs
        ]

        benchmark_runtime = RAGRuntime(
            retrieval_time=mean(
                r.retrieval_time
                for r in runtimes
            ),
            generation_time=mean(
                r.generation_time
                for r in runtimes
            ),
            metric_evaluation_time=mean(
                r.metric_evaluation_time
                for r in runtimes
            ),
        )

        lines.append("\n## Benchmark Runtime\n")

        lines.append(
            "| Avg Retrieval Time | Avg Generation Time | "
            "Avg Metric Evaluation Time | Avg Total Time |"
        )

        lines.append(
            "|:---:|:---:|:---:|:---:|"
        )

        lines.append(
            f"| {benchmark_runtime.retrieval_time:.4f}s | "
            f"{benchmark_runtime.generation_time:.4f}s | "
            f"{benchmark_runtime.metric_evaluation_time:.4f}s | "
            f"{benchmark_runtime.total_time:.4f}s |"
        )

        return "\n".join(lines)