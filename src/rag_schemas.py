from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from statistics import mean
from pydantic import BaseModel, ConfigDict, Field, computed_field
from src.configs import (
    BaseTextGeneratorConfig,
    BaseVectorStoreConfig,
    DoublePassMergingChunkerConfig,
    EmbedderConfig,
    OllamaTextGeneratorConfig,
    QARetrieverConfig,
    QdrantVectorStoreConfig,
    RAGPipelineConfig,
    SourceRetrieverConfig,
)
from src.utility import escape_md, generate_collection_name, load_model, load_models, save_model, save_models

class BaseEvaluationArtifact(
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

class RAGEvaluationResult(BaseEvaluationArtifact):

    sample: RAGEvaluationSample

    metrics: RAGMetrics

    def to_markdown(self) -> str:
        """Generate markdown report for a single RAG evaluation result."""
        lines: list[str] = []

        # ─── Header ───
        lines.append(f"## Question: {self.sample.qa_id}\n")

        # ─── Question & Answers ───
        lines.append("**User Question:**")
        lines.append(f"\n{escape_md(self.sample.question)}\n")
        lines.append("\n**Generated Answer:**")
        lines.append(f"\n{escape_md(self.sample.generated_answer)}\n")
        lines.append("\n**Ground Truth:**")
        lines.append(f"\n{escape_md(self.sample.ground_truth)}\n")

        # ─── Retrieved Chunks ───
        lines.append(f"**Retrieved Chunks ({len(self.sample.retrieved_chunks)}):**\n")
        for i, chunk in enumerate(self.sample.retrieved_chunks, 1):
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            lines.append(f"{i}. {escape_md(preview)}\n")
        lines.append("")  # blank line

        # ─── Metrics Table ───
        lines.append("### Evaluation Scores\n")
        lines.append(
            "| BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | "
            "Context Precision | Context Recall | Faithfulness | Answer Relevancy | Semantic Similarity |"
        )
        lines.append(
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        )
        m = self.metrics
        lines.append(
            f"| {m.bleu:.4f} | {m.rouge1:.4f} | {m.rouge2:.4f} | {m.rougeL:.4f} | "
            f"{m.context_precision:.4f} | {m.context_recall:.4f} | {m.faithfulness:.4f} | "
            f"{m.answer_relevancy:.4f} | {m.semantic_similarity:.4f} |"
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

    source_retriever: SourceRetrieverConfig

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

class RAGEvaluationRun(BaseEvaluationArtifact):

    setup: RAGEvaluationSetup

    results: list[RAGEvaluationResult]

    @computed_field
    @property
    def summary(self) -> RAGEvaluationSummary:

        metrics = [r.metrics for r in self.results]

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

        return "\n".join(lines)

class BenchmarkResult(BaseEvaluationArtifact):

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

        alignment = [
            ":---:"
            if column in metric_columns
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

            lines.append(
                "| " + " | ".join(row) + " |"
            )

            previous_embedder = embedder
            previous_chunker = chunker
            previous_answer_generator = answer_generator
            previous_evaluation_generator = (
                evaluation_generator
            )

        return "\n".join(lines)

