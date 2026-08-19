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
            answer_generator=OllamaTextGeneratorConfig(
                model_name="qwen3.5:9b",
                backend=OllamaBackendConfig(),
                max_new_tokens=2048,
            ),
            evaluation_generator=OllamaTextGeneratorConfig(
                model_name="qwen3.5:9b",
                backend=OllamaBackendConfig(),
                max_new_tokens=4096,
                temperature=0,
                top_p=1
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
    
    rag_experiment_runner.run_multiple_experiments(setups=setups, num_samples=NUM_SAMPLES, save_dir=experiment_dir)

if __name__ == "__main__":
    evaluate_embedder_combination()