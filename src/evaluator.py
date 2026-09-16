from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from itertools import combinations
import math
from pathlib import Path
import time
from typing import Generic, TypeVar
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from src.rag_schemas import (
    BaseArtifact,
    BenchmarkResult,
    RAGEvaluationResult,
    RAGEvaluationRun,
    RAGEvaluationSample,
    RAGEvaluationSetup,
    RAGGenerationRun,
    RAGMetrics,
    RAGPipelineResult,
    RAGRuntime,
    RAGStatisticalEvaluationResult,
    StatisticalExperiment,
    StatisticalTestResult,
)
from src.rag_pipeline import RAGPipeline
from src.utility import logger, clear_gpu

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
    RAGStatisticalEvaluatorConfig,
)

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ResultT = TypeVar(
    "ResultT",
    bound=BaseArtifact,
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

    async def evaluate_async(
        self,
        pipeline_result: RAGPipelineResult,
    ) -> RAGEvaluationResult:

        sample = pipeline_result.sample

        logger.info(
            "Evaluating QA '%s'.",
            sample.qa_id,
        )

        # -------------------------------------------------
        # Metric evaluation
        # -------------------------------------------------

        evaluation_start = time.perf_counter()

        ragas_metrics = await self._compute_ragas_metrics(
            sample,
        )

        classical_metrics = self._compute_bleu_rouge(
            sample,
        )

        metric_evaluation_time = (time.perf_counter() - evaluation_start)

        logger.debug(
            "Metric evaluation for QA '%s' completed in %.4f seconds.",
            sample.qa_id,
            metric_evaluation_time,
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = self._build_metrics(
            ragas_metrics,
            classical_metrics,
        )

        # -------------------------------------------------
        # Runtime
        # -------------------------------------------------

        runtime = RAGRuntime(
            retrieval_time=pipeline_result.retrieval_time,
            generation_time=pipeline_result.generation_time,
            metric_evaluation_time=metric_evaluation_time,
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        return RAGEvaluationResult(
            sample=sample,
            metrics=metrics,
            runtime=runtime,
        )


    async def evaluate_many_async(
        self,
        pipeline_results: list[RAGPipelineResult],
    ) -> list[RAGEvaluationResult]:

        logger.info(
            "Evaluating %d sample(s).",
            len(pipeline_results),
        )

        results: list[RAGEvaluationResult] = []

        total = len(pipeline_results)

        for index, pipeline_result in enumerate(
            pipeline_results,
            start=1,
        ):

            logger.info(
                "[%d/%d] Evaluating QA '%s'.",
                index,
                total,
                pipeline_result.sample.qa_id,
            )

            result = await self.evaluate_async(
                pipeline_result,
            )

            results.append(result)

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
        pipeline_result: RAGPipelineResult,
    ) -> RAGEvaluationResult:

        return asyncio.run(
            self.evaluate_async(
                pipeline_result,
            )
        )

    def evaluate_many(
        self,
        pipeline_results: list[RAGPipelineResult],
    ) -> list[RAGEvaluationResult]:

        return asyncio.run(
            self.evaluate_many_async(
                pipeline_results,
            )
        )

class RAGStatisticalEvaluator(
    BaseEvaluator[RAGStatisticalEvaluationResult],
):

    def __init__(
        self,
        config: RAGStatisticalEvaluatorConfig,
    ):
        super().__init__(config)

        self.config = config

    def _align_results(
        self,
        run_a: RAGEvaluationRun,
        run_b: RAGEvaluationRun,
    ) -> tuple[
        list[RAGEvaluationResult],
        list[RAGEvaluationResult]
        ]:

        results_a = {
            result.sample.qa_id: result
            for result in run_a.results
        }

        results_b = {
            result.sample.qa_id: result
            for result in run_b.results
        }

        common_ids = sorted(
            results_a.keys()
            & results_b.keys()
        )

        if not common_ids:
            raise ValueError(
                "Experiments have no common QA samples."
            )

        aligned_a = [
            results_a[qa_id]
            for qa_id in common_ids
        ]

        aligned_b = [
            results_b[qa_id]
            for qa_id in common_ids
        ]

        return aligned_a, aligned_b

    def _wilcoxon(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> tuple[float, float]:

        if len(values_a) != len(values_b):
            raise ValueError(
                "Wilcoxon samples must have equal length."
            )

        statistic, p_value = wilcoxon(
            values_a,
            values_b,
            alternative="two-sided",
        )

        return (
            float(statistic),
            float(p_value),
        )

    def _apply_correction(
        self,
        results: list[StatisticalTestResult],
    ) -> list[StatisticalTestResult]:

        if not results:
            return results

        if self.config.correction == "none":
            return results

        method_map = {
            "bonferroni": "bonferroni",
            "holm": "holm",
            "fdr_bh": "fdr_bh",
        }

        method = method_map[
            self.config.correction
        ]

        p_values = np.array(
            [
                result.p_value
                for result in results
            ]
        )

        _, adjusted_p_values, _, _ = multipletests(
            p_values,
            alpha=self.config.alpha,
            method=method,
        )

        corrected: list[StatisticalTestResult] = []

        for result, adjusted_p_value in zip(
            results,
            adjusted_p_values,
        ):
            corrected.append(
                result.model_copy(
                    update={
                        "adjusted_p_value": float(
                            adjusted_p_value
                        ),
                        "significant": (
                            adjusted_p_value
                            < self.config.alpha
                        ),
                    },
                )
            )

        return corrected

    def _extract_metric(
        self,
        results: list[RAGEvaluationResult],
        metric: str,
    ) -> list[float]:

        values: list[float] = []

        for result in results:
            value = getattr(
                result.metrics,
                metric,
                None,
            )

            if value is None:
                raise ValueError(
                    f"Metric '{metric}' does not exist "
                    f"in RAGMetrics."
                )

            values.append(value)

        return values

    def _evaluate_pair(
        self,
        run_a: RAGEvaluationRun,
        run_b: RAGEvaluationRun,
        experiment_a: int,
        experiment_b: int,
    ) -> list[StatisticalTestResult]:

        aligned_a, aligned_b = self._align_results(
            run_a,
            run_b,
        )

        results: list[StatisticalTestResult] = []

        for metric in self.config.metrics:

            values_a = self._extract_metric(
                aligned_a,
                metric,
            )

            values_b = self._extract_metric(
                aligned_b,
                metric,
            )

            statistic, p_value = self._wilcoxon(
                values_a,
                values_b,
            )

            results.append(
                StatisticalTestResult(
                    experiment_a=experiment_a,
                    experiment_b=experiment_b,
                    metric=metric,
                    test="wilcoxon",
                    statistic=statistic,
                    p_value=p_value,
                    alpha=self.config.alpha,
                    significant=(
                        p_value
                        < self.config.alpha
                    ),
                )
            )

        return results

    def evaluate(
        self,
        benchmark: BenchmarkResult,
    ) -> RAGStatisticalEvaluationResult:

        logger.info(
            "Running statistical evaluation on %d experiment(s).",
            len(benchmark.runs),
        )

        results: list[StatisticalTestResult] = []

        for (
            index_a,
            index_b,
        ) in combinations(
            range(len(benchmark.runs)),
            2,
        ):
            run_a = benchmark.runs[index_a]
            run_b = benchmark.runs[index_b]

            logger.info(
                "Comparing experiment [%d] vs [%d].",
                index_a + 1,
                index_b + 1,
            )

            results.extend(
                self._evaluate_pair(
                    run_a=run_a,
                    run_b=run_b,
                    experiment_a=index_a + 1,
                    experiment_b=index_b + 1,
                )
            )

        results = self._apply_correction(
            results,
        )

        experiment_mapping = {
            index + 1: StatisticalExperiment(
                embedder=run.setup.embedder.model_name,
                modality=run.setup.source_retriever.modality.display_name,
            )
            for index, run in enumerate(benchmark.runs)
        }

        logger.info("Finished comparing %d pair of experiments", math.comb(len(benchmark.runs), 2))

        return RAGStatisticalEvaluationResult(
            experiment_mapping=experiment_mapping,
            results=results,
        )

    def evaluate_many(
        self,
        benchmarks: list[BenchmarkResult],
    ) -> list[RAGStatisticalEvaluationResult]:

        return [
            self.evaluate(benchmark)
            for benchmark in benchmarks
        ]

class RAGExperimentRunner:

    def __init__(
        self,
        config: RAGEvaluatorConfig,
    ):
        self.config = config

    def generate_experiment(
        self,
        setup: RAGEvaluationSetup,
        num_samples: int | None = None,
    ) -> RAGGenerationRun:

        logger.info(
            "Generating RAG experiment.",
        )

        pipeline = RAGPipeline.build(
            setup,
        )

        results = pipeline.generate_samples(
            limit=num_samples,
        )

        logger.info(
            "Generated %d RAG sample(s).",
            len(results),
        )

        return RAGGenerationRun(
            setup=setup,
            results=results,
        )
    async def evaluate_experiment_async(
        self,
        generation_run: RAGGenerationRun,
        checkpoint_path: str | Path | None = None,
    ) -> RAGEvaluationRun:
        
        logger.info(
            "Evaluating RAG experiment.",
        )

        setup = generation_run.setup

        evaluated_results: list[RAGEvaluationResult] = []

        # Resume from checkpoint if available.
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)

            if checkpoint_path.exists():
                checkpoint = RAGEvaluationRun.from_json(
                    checkpoint_path,
                )

                evaluated_results = checkpoint.results

                logger.info(
                    "Resuming evaluation from checkpoint: "
                    "%d QA sample(s) already evaluated.",
                    len(evaluated_results),
                )
        
        evaluation_generator = OllamaTextGenerator(
            setup.evaluation_generator,
        )

        evaluation_generator.load_pretrained()

        try:
            embedder = Embedder(
                setup.embedder,
            )

            evaluator = RAGEvaluator(
                config=self.config,
                embedder=embedder,
                generator=evaluation_generator,
            )

            remaining_results = generation_run.results[
                len(evaluated_results):
            ]

            for index, pipeline_result in enumerate(
                remaining_results,
                start=len(evaluated_results) + 1,
            ):
                logger.info(
                    "Evaluating QA [%d/%d].",
                    index,
                    len(generation_run.results),
                )

                result = await evaluator.evaluate_async(
                    pipeline_result,
                )

                evaluated_results.append(
                    result,
                )

                # Checkpoint after every successful QA.
                if checkpoint_path is not None:
                    checkpoint = RAGEvaluationRun(
                        setup=setup,
                        results=evaluated_results,
                    )

                    checkpoint.to_json(
                        checkpoint_path,
                    )

            logger.info(
                "Evaluated %d RAG sample(s).",
                len(evaluated_results),
            )

            return RAGEvaluationRun(
                setup=setup,
                results=evaluated_results,
            )

        finally:
            clear_gpu()
        
    def evaluate_experiment(
        self,
        generation_run: RAGGenerationRun,
        checkpoint_path: str | Path | None = None,
    ) -> RAGEvaluationRun:

        return asyncio.run(
            self.evaluate_experiment_async(
                generation_run=generation_run,
                checkpoint_path=checkpoint_path,
            )
        )
    
    def evaluate_experiment_from_json(
        self,
        path: str | Path,
    ) -> RAGEvaluationRun:

        generation_run = RAGGenerationRun.from_json(
            path,
        )

        return self.evaluate_experiment(
            generation_run,
        )

    def generate_multiple_experiments(
        self,
        setups: list[RAGEvaluationSetup],
        num_samples: int | None = None,
        save_dir: str | Path | None = None,
    ) -> list[RAGGenerationRun]:

        save_dir = (
            Path(save_dir)
            if save_dir is not None
            else None
        )

        if save_dir is not None:
            save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        generation_runs: list[RAGGenerationRun] = []

        for index, setup in enumerate(
            setups,
            start=1,
        ):

            logger.info(
                "Generating experiment [%d/%d].",
                index,
                len(setups),
            )

            generation_run = self.generate_experiment(
                setup=setup,
                num_samples=num_samples,
            )

            generation_runs.append(
                generation_run,
            )

            if save_dir is not None:
                generation_run.to_json(
                    save_dir
                    / f"experiment_{index:03d}_details.json"
                )

                generation_run.save_markdown(
                    save_dir
                    / f"experiment_{index:03d}_readable.md"
                )

        return generation_runs
    
    async def evaluate_multiple_experiments_async(
        self,
        generation_runs: list[RAGGenerationRun],
        save_dir: str | Path | None = None,
    ) -> BenchmarkResult:

        save_dir = (
            Path(save_dir)
            if save_dir is not None
            else None
        )

        if save_dir is not None:
            save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        runs: list[RAGEvaluationRun] = []

        for index, generation_run in enumerate(
            generation_runs,
            start=1,
        ):
            logger.info(
                "Evaluating experiment [%d/%d].",
                index,
                len(generation_runs),
            )

            if save_dir is not None:
                final_path = (
                    save_dir
                    / f"experiment_{index:03d}_details.json"
                )

                checkpoint_path = (
                    save_dir
                    / f".checkpoint_experiment_{index:03d}.json"
                )

                # Experiment already completed.
                if final_path.exists():
                    logger.info(
                        "Experiment [%d] already completed. "
                        "Loading saved result.",
                        index,
                    )

                    run = RAGEvaluationRun.from_json(
                        final_path,
                    )

                    runs.append(run)

                    continue

            else:
                checkpoint_path = None

            run = await self.evaluate_experiment_async(
                generation_run=generation_run,
                checkpoint_path=checkpoint_path,
            )

            runs.append(run)

            if save_dir is not None:
                run.to_json(
                    final_path,
                )

                run.save_markdown(
                    save_dir
                    / f"experiment_{index:03d}_readable.md"
                )

                # The experiment completed successfully,
                # so its checkpoint is no longer needed.
                checkpoint_path.unlink(
                    missing_ok=True,
                )

                logger.info(
                    "Experiment [%d] completed successfully.",
                    index,
                )

        benchmark = BenchmarkResult(
            runs=runs,
        )

        if save_dir is not None:
            benchmark.to_json(
                save_dir
                / "experiments_benchmark_details.json"
            )

            benchmark.save_markdown(
                save_dir
                / "experiments_benchmark_summary.md"
            )

        logger.info("All Experiment completed successfully.")

        return benchmark

    def evaluate_multiple_experiments(
        self,
        generation_runs: list[RAGGenerationRun],
        save_dir: str | Path | None = None,
    ) -> BenchmarkResult:

        return asyncio.run(
            self.evaluate_multiple_experiments_async(
                generation_runs=generation_runs,
                save_dir=save_dir,
            )
        )
    
    def evaluate_multiple_experiments_from_dir(
        self,
        directory: str | Path,
        save_dir: str | Path | None = None,
    ) -> BenchmarkResult:

        directory = Path(directory)

        paths = sorted(directory.glob("experiment_*_details.json"))

        generation_runs = [
            RAGGenerationRun.from_json(path)
            for path in paths
        ]

        return self.evaluate_multiple_experiments(
            generation_runs=generation_runs,
            save_dir=save_dir,
        )