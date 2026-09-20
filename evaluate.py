from datetime import datetime
from pathlib import Path

from src.rag_schemas import (
    BenchmarkResult,
)
from src.evaluator import (
    RAGExperimentRunner,
    RAGStatisticalEvaluator,
)

from src.configs import (
    RAGEvaluatorConfig,
    RAGStatisticalEvaluatorConfig
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def evaluate_embedder_combination():

    rag_experiment_runner = RAGExperimentRunner(
        config=RAGEvaluatorConfig(),
    )

    # evaluated_dir = Path("outputs") / f"run-{timestamp}" / "evaluated"
    evaluated_dir = "outputs/run-20260918_142551/evaluated"

    rag_experiment_runner.evaluate_multiple_experiments_from_dir(directory="outputs/run-20260917_153820/generated", save_dir=evaluated_dir)

def evaluate_statistics():
    benchmark_path = "outputs/run-20260912_112917/evaluated/experiments_benchmark_details.json"
    benchmark_result = BenchmarkResult.from_json(benchmark_path)
    statistical_evaluator_config = RAGStatisticalEvaluatorConfig()
    statistical_evaluator = RAGStatisticalEvaluator(statistical_evaluator_config)
    statistical_result = statistical_evaluator.evaluate(benchmark_result)

    statistically_evaluated_dir = Path("outputs") / f"run-{timestamp}" / "statistically_evaluated"

    statistical_evaluator_config.to_json(statistically_evaluated_dir / "statistical_evaluation_config.json")
    statistical_result.to_json(statistically_evaluated_dir / "statistical_summary_details.json")
    statistical_result.save_markdown(statistically_evaluated_dir / "statistical_summary_readable.md")



if __name__ == "__main__":
    evaluate_embedder_combination()
    # evaluate_statistics()