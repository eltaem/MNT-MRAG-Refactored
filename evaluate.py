from datetime import datetime
from pathlib import Path

from src.rag_schemas import (
    BenchmarkResult,
)
from src.evaluator import (
    RAGExperimentRunner,
    RAGMultiStatisticalEvaluator,
    RAGPairStatisticalEvaluator,
)

from src.configs import (
    RAGEvaluatorConfig,
    RAGMultiStatisticalEvaluatorConfig,
    RAGPairStatisticalEvaluatorConfig,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def evaluate_embedder_combination():

    rag_experiment_runner = RAGExperimentRunner(
        config=RAGEvaluatorConfig(),
    )

    # evaluated_dir = Path("outputs") / f"run-{timestamp}" / "evaluated"
    evaluated_dir = "outputs/run-20261123_205131/evaluated"

    rag_experiment_runner.evaluate_multiple_experiments_from_dir(directory="outputs/run-20261123_181909/generated", save_dir=evaluated_dir)

def evaluate_statistics():
    benchmark_path = "outputs/run-20260930_002733/evaluated/experiments_benchmark_details.json"
    benchmark_result = BenchmarkResult.from_json(benchmark_path)
    multi_statistical_evaluator_config = RAGMultiStatisticalEvaluatorConfig()
    multi_statistical_evaluator = RAGMultiStatisticalEvaluator(multi_statistical_evaluator_config)
    multi_statistical_result = multi_statistical_evaluator.evaluate(benchmark_result)
    pair_statistical_evaluator_config = RAGPairStatisticalEvaluatorConfig()
    pair_statistical_evaluator = RAGPairStatisticalEvaluator(pair_statistical_evaluator_config)
    pair_statistical_result = pair_statistical_evaluator.evaluate(benchmark_result)

    statistically_evaluated_dir = Path("outputs") / f"run-{timestamp}" / "statistically_evaluated"

    multi_statistical_evaluator_config.to_json(statistically_evaluated_dir / "multi_statistical_evaluation_config.json")
    pair_statistical_evaluator_config.to_json(statistically_evaluated_dir / "pair_statistical_evaluation_config.json")
    multi_statistical_result.to_json(statistically_evaluated_dir / "multi_statistical_summary_details.json")
    pair_statistical_result.to_json(statistically_evaluated_dir / "pair_statistical_summary_details.json")
    multi_statistical_result.save_markdown(statistically_evaluated_dir / "multi_statistical_summary_readable.md")
    pair_statistical_result.save_markdown(statistically_evaluated_dir / "pair_statistical_summary_readable.md")

def evaluate_merged_statistics():
    benchmark_paths = [
        "outputs/run-20260825_111258_sd_se/summary/experiments_benchmark_details.json", # supporting evidence
        "outputs/run-20260828_235308_sd_sot/summary/experiments_benchmark_details.json", # source of truth
        "outputs/run-20260930_002733_RCTO/evaluated/experiments_benchmark_details.json", # RCTO
    ]

    benchmark_result = BenchmarkResult.merge_from_json(benchmark_paths)
    multi_statistical_evaluator_config = RAGMultiStatisticalEvaluatorConfig()
    multi_statistical_evaluator = RAGMultiStatisticalEvaluator(multi_statistical_evaluator_config)
    multi_statistical_result = multi_statistical_evaluator.evaluate(benchmark_result)
    pair_statistical_evaluator_config = RAGPairStatisticalEvaluatorConfig()
    pair_statistical_evaluator = RAGPairStatisticalEvaluator(pair_statistical_evaluator_config)
    pair_statistical_result = pair_statistical_evaluator.evaluate(benchmark_result)

    statistically_evaluated_dir = Path("outputs") / f"run-{timestamp}" / "statistically_evaluated"

    multi_statistical_evaluator_config.to_json(statistically_evaluated_dir / "multi_statistical_evaluation_config.json")
    pair_statistical_evaluator_config.to_json(statistically_evaluated_dir / "pair_statistical_evaluation_config.json")
    multi_statistical_result.to_json(statistically_evaluated_dir / "multi_statistical_summary_details.json")
    pair_statistical_result.to_json(statistically_evaluated_dir / "pair_statistical_summary_details.json")
    multi_statistical_result.save_markdown(statistically_evaluated_dir / "multi_statistical_summary_readable.md")
    pair_statistical_result.save_markdown(statistically_evaluated_dir / "pair_statistical_summary_readable.md")
    
if __name__ == "__main__":
    evaluate_embedder_combination()
    # evaluate_statistics()