from datetime import datetime
from pathlib import Path

from src.evaluator import (
    RAGEvaluationSetup,
    RAGExperimentRunner,
)

from src.configs import (
    RAGEvaluatorConfig,
    EmbedderConfig,
    DoublePassMergingChunkerConfig,
    QdrantVectorStoreConfig,
    QdrantConfig,
    RAGPipelineConfig,
    OllamaBackendConfig,
    OllamaTextGeneratorConfig,
    SourceRetrieverConfig,
    QARetrieverConfig,
)

from src.utility import Modality

# =====================================================
# Experiment Configurations
# =====================================================

EMBEDDING_NAMES = [
        "nvidia/llama-embed-nemotron-8b",
        "Octen/Octen-Embedding-8B",
        "codefuse-ai/F2LLM-v2-8B",
    ]

MODALITIES = [
    Modality.TEXT,
    Modality.TEXT_TABLE,
    Modality.TEXT_IMAGE,
    Modality.TEXT_TABLE_IMAGE,
]

NUM_SAMPLES = 50

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = Path("outputs") / f"run-{timestamp}"

def evaluate_embedder_combination():

    setups = [
        RAGEvaluationSetup(
            embedder=EmbedderConfig(
                model_name=embedding_name,
            ),
            chunker=DoublePassMergingChunkerConfig(),
            vector_store=QdrantVectorStoreConfig(
                backend=QdrantConfig(),
            ),
            rag=RAGPipelineConfig(),
            generator=OllamaTextGeneratorConfig(
                model_name="qwen3.5:9b",
                backend=OllamaBackendConfig(),
                num_predict=4096,
            ),
            source_retriever=SourceRetrieverConfig(
                modality=modality,
            ),
            qa_retriever=QARetrieverConfig(),
        )
        for embedding_name in EMBEDDING_NAMES
        for modality in MODALITIES
    ]

    rag_experiment_runner = RAGExperimentRunner(
        config=RAGEvaluatorConfig(),
    )
    
    experiments = rag_experiment_runner.run_multiple_experiments(setups=setups, num_samples=NUM_SAMPLES)
    experiments.to_json(experiment_dir / "multiple_experiment_result.json")
    experiments.save_markdown(experiment_dir / "multiple_experiment_result_md.md")

if __name__ == "__main__":
    evaluate_embedder_combination()