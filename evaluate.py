from datetime import datetime
from pathlib import Path

from src.evaluator import (
    RAGExperimentRunner,
)

from src.configs import (
    RAGEvaluatorConfig,
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
evaluated_dir = Path("outputs") / f"run-{timestamp}" / "evaluated"

def evaluate_embedder_combination():

    rag_experiment_runner = RAGExperimentRunner(
        config=RAGEvaluatorConfig(),
    )
    
    rag_experiment_runner.evaluate_multiple_experiments_from_dir(directory="", save_dir=evaluated_dir)

if __name__ == "__main__":
    evaluate_embedder_combination()