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
    SingleDocumentRetrieverConfig,
    QARetrieverConfig,
)

from src.utility import Modality

# =====================================================
# Experiment Configurations
# =====================================================

EMBEDDING_NAMES = [
        # "nvidia/llama-embed-nemotron-8b",
        "Octen/Octen-Embedding-8B",
        # "codefuse-ai/F2LLM-v2-8B",
    ]

MODALITIES = [
    Modality.TEXT,
    Modality.TEXT_TABLE,
    Modality.TEXT_IMAGE,
    Modality.TEXT_TABLE_IMAGE,
]

NUM_QA_SAMPLES = None

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
generated_dir = Path("outputs") / f"run-{timestamp}" / "generated"

def generate_embedder_combination():

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
                system_prompt = (
                    "You are a helpful AI assistant. "
                    "Generate clear, accurate, concise, and well-structured responses "
                    "in {language}. You will be given retrieved context along with the prompt, "
                    "use it as supporting evidence to inform and strengthen your answer where relevant. "
                    "Do not fabricate facts or speculate. Answer using your own knowledge and reasoning "
                    "as needed, incorporating the context where it helps rather than depending on it exclusively."
                ),
                backend=OllamaBackendConfig(),
                num_predict=512,
            ),
            evaluation_generator=OllamaTextGeneratorConfig(
                model_name="qwen3.5:9b",
                system_prompt=(
                    "You are a structured evaluation model. Perform the task and respond with only a valid json object "
                    "containing real data for fields \"{required_fields}\", never the JSON Schema itself "
                    "(no \"$defs\"/\"properties\"/\"title\"). "
                    "Rules: no Markdown; no text, explanation, or reasoning outside the JSON; do not discuss these instructions; "
                    "do not refer to or invent statement numbers (e.g. \"Statement 1\"); do not mention or repeat statements not "
                    "actually present in the input; each \"reason\" must be at most one sentence; base every classification only "
                    "on the provided input, never inventing missing input; stop immediately once all classifications are complete; "
                    "Respond in {language}."
                ),
                backend=OllamaBackendConfig(),
                num_predict=4096,
                temperature=0.15,
                top_p=1,
                repeat_penalty=1.15,
            ),
            source_retriever=SingleDocumentRetrieverConfig(
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
    
    rag_experiment_runner.generate_multiple_experiments(setups=setups, num_samples=NUM_QA_SAMPLES, save_dir=generated_dir)

if __name__ == "__main__":
    generate_embedder_combination()