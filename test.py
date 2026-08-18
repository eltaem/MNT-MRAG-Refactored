import asyncio
from pathlib import Path
from datetime import datetime

def test_image_captioning():
    from src.configs import OllamaImageCaptioningConfig
    from src.models import OllamaImageCaptioningModel
    # from src.configs import TransformerImageCaptioningConfig
    # from src.models import TransformerImageCaptioningModel
    captioning_config = OllamaImageCaptioningConfig(model_name="qwen3.5:9b", device="cuda", max_new_tokens=512)
    captioning_model = OllamaImageCaptioningModel(captioning_config)
    captioning_model.load_pretrained()
    caption1 = captioning_model.caption(input_data="data/images/0c0ea51a-cd46-5f4c-8d04-61c541aca26f/page_013/image_001_b173d73a.png")
    caption2 = captioning_model.caption(input_data="data/images/0c0ea51a-cd46-5f4c-8d04-61c541aca26f/page_013/image_002_c4c1ab7b.png")

    print(caption1)
    print(caption2)

def test_table_captioning():
    from src.models import OllamaTableCaptioningModel
    from src.configs import OllamaTableCaptioningConfig
    table = [
            [
            "NAMA : An. NCN\nNAMA KELUARGA: R\nUSIA : 2 th\nBANGSAL : RII BED 10\nNO. RM : 1230XXXX\nJENIS KELAMIN : Perempuan",
            None,
            None,
            "Tanggal",
            ""
            ],
            [
            None,
            None,
            None,
            "TTD\npetugas",
            ""
            ],
            [
            None,
            None,
            None,
            "BB",
            "9 kg"
            ],
            [
            None,
            None,
            None,
            "TB",
            "79,3 cm"
            ],
            [
            None,
            None,
            None,
            "BMI",
            "14,3 kg/m2"
            ],
            [
            "LANGKAH\n1",
            "Apakah BMI berada di\nbawah acuan?",
            "Tidak",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya (LILA kurang dari\nacuan)",
            "1",
            "1"
            ],
            [
            "LANGKAH\n2",
            "Apakah akhir – akhir\nini anak mengalami\npenurunan berat\nbadan ?",
            "Tidak",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya\n- Kehilangan BB\nyang tidak\ndiharapkan\n- Baju terasa\nlonggar\n- Penambahan BB\nyang tidak\nadekuat (jika < 2\ntahun)",
            "1",
            "1"
            ],
            [
            "LANGKAH\n3",
            "Apakah satu minggu\nterakhir anak\nmengalami penurunan\nasupan makan?",
            "Tidak\n(asupan makan\nseperti biasa)",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya\n(Mengalami\npenurunan asupan\nmakan untuk satu\nminggu terakhir)",
            "1",
            "1"
            ],
            [
            None,
            None,
            "Ya\n(tidak ada asupan\natau asupan sangat\nsedikit untuk satu\nminggu terakhir)",
            "2",
            ""
            ],
            [
            "LANGKAH\n4",
            "Akankah kebutuhan\ngizi anak dipengaruhi\noleh kondisi anak\n(untuk kurang lebih\nsatu minggu ke\ndepan)?",
            "Tidak",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya\n(untuk satu minggu ke\ndepan) :\n- Mengalami\npenurunan\nasupan\ndan/atau\n- Mengalami\npeningkatan\nkebutuhan\ndan/atau\n- Mengalami\npeningkatan\nkehilangan",
            "1",
            "1"
            ],
            [
            None,
            None,
            "Ya",
            "2",
            ""
            ],
            [
            "",
            "",
            "Tidak ada asupan\n(atau asupan sangat\nsedikit) untuk satu\nminggu ke depan",
            "",
            ""
            ],
            [
            "LANGKAH\n5",
            "Jumlahkan skor\nkeseluruhan (total dari\nlangkah 1 hingga\nlangkah 4)",
            "TOTAL SKOR PYMS",
            None,
            "4"
            ]
        ]
    captioning_config = OllamaTableCaptioningConfig(model_name="qwen3.5:9b", device="cuda", max_new_tokens=512)
    captioning_model = OllamaTableCaptioningModel(captioning_config)
    captioning_model.load_pretrained()
    caption = captioning_model.caption(input_data=table)

    print(caption)
    
def check_len():
    text = "Tabel tersebut memuat rekaman klinis seorang anak perempuan 2 tahun bernama An. NCN (RM 1230XXXX, bangsal RII BED 10) dengan berat badan 9 kg, tinggi badan 79,3 cm"
    print(len(text))

def test_hashing():
    from src.utility import generate_document_id
    a_id = generate_document_id("data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf")
    b_id = generate_document_id("data/PDFs/ANAK DENGAN TB PARU , PNEUMONIA , GIZI BURUK MARASMUS.pdf")
    print(a_id)
    print(b_id)
    print(a_id == b_id)

def test_image_extractor():
    from src.configs import ImageExtractorConfig
    from src.extractors import ImageExtractor
    from src.utility import generate_document_id
    import fitz
    config = ImageExtractorConfig(image_dir="data/images")
    extracttt = ImageExtractor(config)
    doc_name = "data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    doc = fitz.open(doc_name)
    page_number = 33
    page = doc.load_page(page_number)
    print(generate_document_id(doc_name))
    nodes = extracttt.extract(page, generate_document_id(doc_name))
    for node in nodes:
        print(node)

def test_text_extractor():
    from src.configs import TextExtractorConfig
    from src.extractors import TextExtractor
    from src.utility import generate_document_id
    import fitz
    config = TextExtractorConfig()
    extracttt = TextExtractor(config)
    doc_name = "data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    doc = fitz.open(doc_name)
    page_number = 33
    page = doc.load_page(page_number)
    print(generate_document_id(doc_name))
    nodes = extracttt.extract(page, generate_document_id(doc_name))
    for node in nodes:
        print(node)

def test_table_extractor():
    from src.configs import TableExtractorConfig
    from src.extractors import TableExtractor
    from src.utility import generate_document_id
    import fitz
    config = TableExtractorConfig()
    extracttt = TableExtractor(config)
    doc_name = "data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    doc = fitz.open(doc_name)
    page_number = 29
    page = doc.load_page(page_number)
    print(generate_document_id(doc_name))
    nodes = extracttt.extract(page, generate_document_id(doc_name))
    for node in nodes:
        print(node)

def test_pdf_extractor():
    from src.configs import PDFExtractorConfig
    from src.extractors import PDFExtractor
    config = PDFExtractorConfig()

    extractor = PDFExtractor(config)

    nodes = extractor.extract(
        "data/test/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_debug_path = Path("outputs") / f"run-{timestamp}" / "debug" / "nodes.json"
    PDFExtractor.to_json(nodes, pdf_debug_path)

def test_pdfs_extractor():
    from src.configs import PDFExtractorConfig
    from src.extractors import PDFExtractor
    config = PDFExtractorConfig()
    extractor = PDFExtractor(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_debug_dir = Path("outputs") / f"run-{timestamp}" / "debug"
    documents = extractor.extract_many("data/test")
    for i, (path, nodes) in enumerate(documents.items()):
        print(path, len(nodes))
        PDFExtractor.to_json(nodes, pdf_debug_dir / "many_documents" / f"document_{i}.json")

def test_pdf_processor(path:Path):
    from src.processors import RemoveBeforePatternProcessor, TextMerger
    from pdf_processor import PDFProcessor
    from src.configs import (
        PDFExtractorConfig,
        RemoveBeforePatternProcessorConfig,
        TextMergerConfig,
        DEFAULT_FRONT_MATTER_PATTERNS,
        TextMergeMode
    )
    from src.extractors import PDFExtractor
    config = PDFExtractorConfig()
    extractor = PDFExtractor(config)
    remove_config = RemoveBeforePatternProcessorConfig(
        patterns=DEFAULT_FRONT_MATTER_PATTERNS
    )
    merger_config = TextMergerConfig(
        mode=TextMergeMode.DOCUMENT
    )
    processor = PDFProcessor(
        extractor=extractor,
        processors=[
            RemoveBeforePatternProcessor(
                remove_config,
            ),
            TextMerger(
                merger_config,
            ),
        ],
    )
    output_path = path / "nodes.json"
    nodes = processor.process_pdf(
        "data/test/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf",
    )

    PDFProcessor.to_json(
        nodes,
        output_path,
    )

def test_pdfs_processor(path:Path):
    from src.processors import RemoveBeforePatternProcessor, TextMerger
    from pdf_processor import PDFProcessor
    from src.configs import (
        PDFExtractorConfig,
        PDFProcessorConfig,
        RemoveBeforePatternProcessorConfig,
        TextMergerConfig,
        DEFAULT_FRONT_MATTER_PATTERNS,
        TextMergeMode
    )
    from src.extractors import PDFExtractor
    processor_config = PDFProcessorConfig()
    config = PDFExtractorConfig()
    extractor = PDFExtractor(config)
    remove_config = RemoveBeforePatternProcessorConfig(
        patterns=DEFAULT_FRONT_MATTER_PATTERNS
    )
    merger_config = TextMergerConfig(
        mode=TextMergeMode.DOCUMENT
    )
    processor = PDFProcessor(
        config=processor_config,
        extractor=extractor,
        processors=[
            RemoveBeforePatternProcessor(
                remove_config,
            ),
            TextMerger(
                merger_config,
            ),
        ],
    )
    output_path = path / "nodes.json"
    nodes = processor.process_pdfs_dir("data/test")

    PDFProcessor.to_json(
        nodes,
        output_path,
    )

async def test_apdfs_processor(path:Path):
    from src.processors import RemoveBeforePatternProcessor, TextMerger
    from pdf_processor import PDFProcessor
    from src.configs import (
        PDFExtractorConfig,
        PDFProcessorConfig,
        RemoveBeforePatternProcessorConfig,
        TextMergerConfig,
        DEFAULT_FRONT_MATTER_PATTERNS,
        TextMergeMode,
        OllamaImageCaptioningConfig,
        OllamaTableCaptioningConfig,
        ImageCaptionEnricherConfig,
        TableCaptionEnricherConfig

    )
    from src.extractors import PDFExtractor
    from src.models import (
        OllamaImageCaptioningModel,
        OllamaTableCaptioningModel
    )
    from src.enrichers import (
        ImageCaptionEnricher,
        TableCaptionEnricher
    )
    processor_config = PDFProcessorConfig()
    extractor_config = PDFExtractorConfig()
    extractor = PDFExtractor(extractor_config)
    remove_config = RemoveBeforePatternProcessorConfig(
        patterns=DEFAULT_FRONT_MATTER_PATTERNS
    )
    merger_config = TextMergerConfig(
        mode=TextMergeMode.DOCUMENT
    )
    image_captioning_config = OllamaImageCaptioningConfig(model_name="qwen3.5:9b", device="cuda", max_new_tokens=512)
    image_captioning_model = OllamaImageCaptioningModel(image_captioning_config)
    image_captioning_model.load_pretrained()
    table_captioning_config = OllamaTableCaptioningConfig(model_name="qwen3.5:9b", device="cuda", max_new_tokens=512)
    table_captioning_model = OllamaTableCaptioningModel(table_captioning_config)
    table_captioning_model.load_pretrained()
    image_enricher_config = ImageCaptionEnricherConfig(batch_size=2)
    table_enricher_config = TableCaptionEnricherConfig(batch_size=2)

    processor = PDFProcessor(
        config=processor_config,
        extractor=extractor,
        processors=[
            RemoveBeforePatternProcessor(
                remove_config,
            ),
            TextMerger(
                merger_config,
            ),
        ],
        enrichers= [
            ImageCaptionEnricher(
                image_enricher_config,
                image_captioning_model
            ),
            TableCaptionEnricher(
                table_enricher_config,
                table_captioning_model
            )
        ]
    )
    output_path = path / "nodes.json"
    nodes = await processor.aprocess_pdfs_dir("data/test")

    PDFProcessor.to_json(
        nodes,
        output_path,
    )

def test_dpmc(path: Path):
    from pdf_processor import PDFProcessor
    from src.configs import EmbedderConfig, DoublePassMergingChunkerConfig
    from src.embedder import Embedder
    from src.chunker import DoublePassMergingChunker
    dpmc_conf = DoublePassMergingChunkerConfig()
    emb_conf = EmbedderConfig(model_name="Octen/Octen-Embedding-8B", target_dimension=768)
    embedder = Embedder(emb_conf)
    dpmc = DoublePassMergingChunker(dpmc_conf, embedder)
    nodes_path = Path("outputs/run-20260630_145750/debug/nodes.json")
    nodes = PDFProcessor.from_json(nodes_path)
    output_path = path / "chunks.json"
    chunks = dpmc.chunk(nodes)
    dpmc.to_json(chunks, output_path)

    
def runtime() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_debug_dir = Path("outputs") / f"run-{timestamp}" / "debug"
    return pdf_debug_dir

def test_src_retriever():
    from src.retriever import SourceRetriever
    from src.configs import QdrantVectorStoreConfig, SourceRetrieverConfig
    from src.vector_store import QdrantVectorStore
    from src.utility import Modality
    vs_conf = QdrantVectorStoreConfig()
    vector_store = QdrantVectorStore(vs_conf)
    retriever = SourceRetriever(SourceRetrieverConfig(modality=Modality.TEXT_TABLE), vector_store, "src-dpmc-codefuse-ai-F2LLM-v2-8B")
    chunks = retriever.scroll(10)
    for chunk in chunks:
        print("="*50)
        print(f"Node type: {chunk.node_type}")
        print(chunk.text)
        print("="*50)

def test_qa_retriever(path: Path):
    from src.retriever import QARetriever
    from src.configs import QdrantVectorStoreConfig, QARetrieverConfig
    from src.vector_store import QdrantVectorStore
    from qa_parser import QARecord
    vs_conf = QdrantVectorStoreConfig()
    vector_store = QdrantVectorStore(vs_conf)
    retriever = QARetriever(QARetrieverConfig(), vector_store, "qa-codefuse-ai-F2LLM-v2-8B")
    qa_records = retriever.scroll(2)
    output_path = path / "records.json"
    QARecord.to_json(qa_records, output_path)
    for qa_record in qa_records:
        print("="*50)
        print("Question                 : ", qa_record.qa.question)
        print("Answer (Ground Truth)    : ", qa_record.qa.answer)
        print("="*50)

def test_rag_pipeline(path:Path):
    from src.configs import (
        RAGPipelineConfig,
        QARetrieverConfig,
        QdrantVectorStoreConfig,
        SourceRetrieverConfig,
        OllamaTextGeneratorConfig,
    )
    from src.rag_pipeline import RAGPipeline, RAGEvaluationSample
    from src.retriever import QARetriever, SourceRetriever
    from src.vector_store import QdrantVectorStore
    from src.models import OllamaTextGenerator
    rag_pipeline_config = RAGPipelineConfig()
    qa_r_conf = QARetrieverConfig()
    vstore_conf = QdrantVectorStoreConfig()
    vector_store = QdrantVectorStore(vstore_conf)
    qa_retriever = QARetriever(qa_r_conf, vector_store, "qa-codefuse-ai-F2LLM-v2-8B")
    src_r_conf = SourceRetrieverConfig()
    src_retriever = SourceRetriever(src_r_conf, vector_store, "src-dpmc-codefuse-ai-F2LLM-v2-8B")
    tgen_conf = OllamaTextGeneratorConfig(model_name="qwen3.5:9b", max_new_tokens=512)
    text_generator = OllamaTextGenerator(tgen_conf)
    text_generator.load_pretrained()
    rag_pipeline = RAGPipeline(
        rag_pipeline_config,
        qa_retriever,
        src_retriever,
        text_generator
    )

    samples = rag_pipeline.generate_samples(50)

    output_path = path / "eval_sample.json"
    RAGEvaluationSample.to_json(samples, output_path)

def test_rag_evaluator(path:Path):
    from src.evaluator import (
        RAGEvaluationSample,
        RAGEvaluator,
    )
    from src.configs import (
        RAGEvaluatorConfig,
        EmbedderConfig,
        OllamaBackendConfig,
        OllamaTextGeneratorConfig
    )
    from src.embedder import Embedder
    from src.models import OllamaTextGenerator
    from src.utility import save_models
    samples = RAGEvaluationSample.from_json("outputs/run-20260713_082634/debug/eval_sample.json")
    embedder = Embedder(EmbedderConfig(model_name="codefuse-ai/F2LLM-v2-8B"))
    generator = OllamaTextGenerator(OllamaTextGeneratorConfig(model_name="qwen3.5:9b", max_new_tokens=2048, backend=OllamaBackendConfig()))
    generator.load_pretrained()
    evaluator = RAGEvaluator(RAGEvaluatorConfig(), embedder, generator)
    results = evaluator.evaluate_many(samples)
    save_models(results, path / "evaluation_result.json")

def test_single_experiment(path:Path):
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

    setup = RAGEvaluationSetup(
        embedder=EmbedderConfig(
            model_name="Octen/Octen-Embedding-8B"
        ),
        chunker=DoublePassMergingChunkerConfig(),
        vector_store=QdrantVectorStoreConfig(
            backend=QdrantConfig()
        ),
        rag=RAGPipelineConfig(),
        generator=OllamaTextGeneratorConfig(
            model_name="qwen3.5:9b",
            backend=OllamaBackendConfig(),
            max_new_tokens=4096,
        ),
        source_retriever=SourceRetrieverConfig(
            modality=Modality.TEXT
        ),
        qa_retriever=QARetrieverConfig()
    )

    rag_experiment_runner = RAGExperimentRunner(
        config=RAGEvaluatorConfig(),
    )

    experiment = rag_experiment_runner.run_single_experiment(setup=setup, num_samples=2)
    experiment.to_json(path / "one_experiment_result.json")
    experiment.save_markdown(path / "one_experiment_result_md.md")

def test_multiple_experiments(path:Path):

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

    embedding_names = [
        "nvidia/llama-embed-nemotron-8b",
        "Octen/Octen-Embedding-8B",
        "codefuse-ai/F2LLM-v2-8B",
    ]

    modalities = [
        Modality.TEXT,
        Modality.TEXT_TABLE,
        Modality.TEXT_IMAGE,
        Modality.TEXT_TABLE_IMAGE,
    ]

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
                max_new_tokens=4096,
            ),
            source_retriever=SourceRetrieverConfig(
                modality=modality,
            ),
            qa_retriever=QARetrieverConfig(),
        )
        for embedding_name in embedding_names
        for modality in modalities
    ]

    rag_experiment_runner = RAGExperimentRunner(
        config=RAGEvaluatorConfig(),
    )
    
    experiments = rag_experiment_runner.run_multiple_experiments(setups=setups, num_samples=2)
    experiments.to_json(path / "multiple_experiment_result.json")
    experiments.save_markdown(path / "multiple_experiment_result_md.md")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Start testing, time: {timestamp}")
    # test_qa_retriever(runtime())
    # test_dpmc(runtime())
    # asyncio.run(test_apdfs_processor(runtime()))
    # test_rag_pipeline(runtime())
    # test_rag_evaluator(runtime())
    test_multiple_experiments(runtime())