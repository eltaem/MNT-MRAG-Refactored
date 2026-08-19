from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Generic, TypeVar
import sacrebleu
from rouge_score import rouge_scorer
from src.rag_schemas import (
    BaseEvaluationArtifact,
    BenchmarkResult,
    RAGEvaluationResult,
    RAGEvaluationRun,
    RAGEvaluationSample,
    RAGEvaluationSetup,
    RAGMetrics,
)
from src.rag_pipeline import RAGPipeline
from src.utility import logger

from src.ragas_adapter import RAGASLLMAdapter
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    SemanticSimilarity
)

from src.models import BaseTextGeneratorModel, OllamaTextGenerator
from src.embedder import Embedder
from src.configs import (
    BaseEvaluatorConfig,
    RAGEvaluatorConfig,
)

ResultT = TypeVar(
    "ResultT",
    bound=BaseEvaluationArtifact,
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

        logger.debug("Computing Faithfulness for QA sample '%s'...", sample.qa_id)
        faithfulness = await self.faithfulness.ascore(
            user_input=sample.question,
            response=sample.generated_answer,
            retrieved_contexts=sample.retrieved_chunks,
        )

        logger.debug("Computing Answer Relevancy for QA sample '%s'...", sample.qa_id)
        answer_relevancy = await self.answer_relevancy.ascore(
            user_input=sample.question,
            response=sample.generated_answer,
        )

        logger.debug("Computing Semantic Similarity for QA sample '%s'...", sample.qa_id)
        semantic_similarity = await self.semantic_similarity.ascore(
            reference=sample.ground_truth,
            response=sample.generated_answer,
        )

        logger.debug("Computing Context Precision for QA sample '%s'...", sample.qa_id)
        context_precision = await self.context_precision.ascore(
            user_input=sample.question,
            reference=sample.ground_truth,
            retrieved_contexts=sample.retrieved_chunks,
        )

        logger.debug("Computing Context Recall for QA sample '%s'...", sample.qa_id)
        context_recall = await self.context_recall.ascore(
            user_input=sample.question,
            retrieved_contexts=sample.retrieved_chunks,
            reference=sample.ground_truth,
        )

        logger.debug("Finished RAGAS Metrics for QA sample '%s'", sample.qa_id)
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
    # Internal Async API
    # =====================================================

    async def _evaluate(
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

        # -------------------------------------------------
        # RAGAS
        # -------------------------------------------------

        logger.debug(
            "Running RAGAS metrics for QA '%s'.",
            sample.qa_id,
        )

        ragas_metrics = await self._compute_ragas_metrics(
            sample,
        )

        logger.debug(
            "RAGAS metrics for QA '%s': %s",
            sample.qa_id,
            ragas_metrics,
        )

        # -------------------------------------------------
        # Classical metrics
        # -------------------------------------------------

        logger.debug(
            "Running BLEU and ROUGE for QA '%s'.",
            sample.qa_id,
        )

        classical_metrics = self._compute_bleu_rouge(
            sample,
        )

        logger.debug(
            "Classical metrics for QA '%s': %s",
            sample.qa_id,
            classical_metrics,
        )

        # -------------------------------------------------
        # Build result
        # -------------------------------------------------

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


    async def _evaluate_many(
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

            result = await self._evaluate(
                sample,
            )

            results.append(
                result,
            )

        logger.info(
            "Finished evaluating %d sample(s).",
            len(results),
        )

        return results


    # =====================================================
    # Public API
    # =====================================================

    def evaluate(
        self,
        sample: RAGEvaluationSample,
    ) -> RAGEvaluationResult:

        return asyncio.run(
            self._evaluate(
                sample,
            )
        )

    def evaluate_many(
        self,
        samples: list[RAGEvaluationSample],
    ) -> list[RAGEvaluationResult]:

        return asyncio.run(
            self._evaluate_many(
                samples,
            )
        )

class RAGExperimentRunner:

    def __init__(
        self,
        config: RAGEvaluatorConfig,
    ):
        self.config = config

    def run_single_experiment(
        self,
        setup: RAGEvaluationSetup,
        num_samples: int | None = None,
    ) -> RAGEvaluationRun:

        pipeline = RAGPipeline.build(setup)

        evaluation_generator = OllamaTextGenerator(
            setup.evaluation_generator
        )

        evaluation_generator.load_pretrained()

        evaluator = RAGEvaluator(
            config=self.config,
            embedder=pipeline.embedder,
            generator=evaluation_generator,
        )

        samples = pipeline.generate_samples(
            limit=num_samples,
        )

        results = evaluator.evaluate_many(
            samples,
        )

        return RAGEvaluationRun(
            setup=setup,
            results=results,
        )

    def run_multiple_experiments(
        self,
        setups: list[RAGEvaluationSetup],
        num_samples: int | None = None,
        save_dir: str | Path | None = None,
    ) -> BenchmarkResult:
        
        save_dir = Path(save_dir) if save_dir is not None else None

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True,)

        runs: list[RAGEvaluationRun] = []

        for index, setup in enumerate(setups, start=1,):

            run = self.run_single_experiment(
                setup=setup,
                num_samples=num_samples,
            )

            runs.append(run)

            if save_dir is not None:
                md_path = save_dir / "individual_experiments" / f"experiment_{index}.md"

                run.save_markdown(md_path)

        benchmark = BenchmarkResult(runs=runs)

        if save_dir is not None:
            benchmark.to_json(save_dir / "summary" / "experiment_benchmark_result.json")
            benchmark.save_markdown(save_dir / "summary" / "experiment_benchmark_result_table.md")

        return benchmark