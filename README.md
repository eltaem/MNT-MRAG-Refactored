# MNT-MRAG-Refactored

This repository implements a research-oriented medical nutrition therapy (MNT) retrieval-augmented generation (RAG) pipeline. It is designed to process PDF-based clinical nutrition documents, convert them into retrieval-ready chunks, index them in a vector store, and benchmark different embedding models and retrieval setups on domain-specific QA tasks.

The code is organized as a modular experimental pipeline rather than a user-facing web app. Its goal is to let researchers compare how different embeddings, chunking strategies, and source-document modalities affect answer quality on medical nutrition questions.

## Research purpose

The project evaluates multiple embedding backbones and document modalities for a nutrition-focused QA setting:

- `nvidia/llama-embed-nemotron-8b`
- `Octen/Octen-Embedding-8B`
- `codefuse-ai/F2LLM-v2-8B`

These are combined with retrieval settings for:

- `TEXT`
- `TEXT_TABLE`
- `TEXT_IMAGE`
- `TEXT_TABLE_IMAGE`

The experimental setup is defined in `generate.py`, and experiment outputs are saved under timestamped folders in `outputs/`.

## High-level pipeline

The end-to-end workflow is:

1. Read raw PDFs from `data/PDFs/`.
2. Extract text, table, and image content from the PDFs.
3. Clean and normalize the extracted nodes.
4. Merge and enrich document content with image/table captioning where relevant.
5. Chunk the processed documents into semantic retrieval units.
6. Embed chunks and index them in Qdrant.
7. Parse the QA dataset and embed QA records for retrieval.
8. Retrieve relevant chunks and similar QA examples for a question.
9. Build a prompt with retrieved context and ask an Ollama-backed model to answer.
10. Evaluate the answer with classical and RAGAS-based metrics.
11. Save experiment results and statistical comparisons.

## Repository structure

```text
.
├── README.md
├── Makefile
├── main.py
├── generate.py
├── evaluate.py
├── pdf_processor.py
├── chunk_processor.py
├── qa_parser.py
├── qa_indexer.py
├── src_indexer.py
├── src/
│   ├── chunker.py
│   ├── configs.py
│   ├── embedder.py
│   ├── evaluator.py
│   ├── enrichers.py
│   ├── extractors.py
│   ├── indexer.py
│   ├── models.py
│   ├── nodes.py
│   ├── processors.py
│   ├── rag_pipeline.py
│   ├── rag_schemas.py
│   ├── ragas_adapter.py
│   ├── record.py
│   ├── retriever.py
│   ├── utility.py
│   ├── vector_store.py
│   └── ...
├── scripts/
│   ├── dependencies.sh
│   ├── evaluator.sh
│   ├── generator.sh
│   ├── kill.sh
│   ├── ollama.sh
│   ├── processor.sh
│   ├── qdrant.sh
│   ├── qa-indexer.sh
│   └── src-indexer.sh
├── data/
│   ├── PDFs/
│   ├── QA/
│   ├── chunks/
│   ├── images/
│   ├── parsed/
│   └── ...
├── outputs/
│   └── run-<timestamp>/
│       ├── generated/
│       ├── evaluated/
│       └── statistically_evaluated/
├── .env
├── .gitignore
└── qdrant/
```

## Required services and packages

This project expects the following:

- Python 3.10+
- Qdrant running locally or on a reachable host
- Ollama running locally for the generation model(s)
- PyTorch and transformer-compatible embedding backends
- Hugging Face / sentence-transformer style embedding models
- Pydantic
- pandas
- Python-dotenv
- PyMuPDF (`fitz`)
- Qdrant client
- scikit/NumPy ecosystem for metrics
- RAGAS, ROUGE-score, sacrebleu, SciPy, statsmodels

## Environment setup

The code reads `PROJECT_ROOT` from the environment in `src/configs.py`.

```bash
export PROJECT_ROOT="/absolute/path/to/MNT-MRAG-Refactored"
```

Then bring up the supporting services:

```bash
make qdrant
make ollama
```

or directly:

```bash
bash scripts/qdrant.sh
bash scripts/ollama.sh
```

If you want both in one command:

```bash
bash scripts/dependencies.sh
```

## Data layout

### Source documents

Put the raw PDF corpus under:

```text
data/PDFs/
```

### Parsed documents

After PDF processing, extracted document objects are stored under:

```text
data/parsed/
```

### Chunked documents

Chunked and embedded document records are written under:

```text
data/chunks/
```

The project builds chunk directories by chunking method and embedding model.

### QA data

The raw QA dataset is expected at:

```text
data/QA/indonutriqa.json
```

The parser (`qa_parser.py`) normalizes the source fields into structured QA items and writes:

```text
data/QA/QA_dataset.json
```

and:

```text
data/QA/QA_dataset.csv
```

The QA embeddings are also cached under:

```text
data/QA/embeddings/
```

## Processing workflow

### 1) Parse and normalize the QA dataset

```bash
python qa_parser.py
```

or:

```bash
make pqa
```

This parses the raw QA JSON and prepares a cleaned dataset for indexing and evaluation.

### 2) Process PDFs into structured nodes

```bash
python pdf_processor.py
```

or:

```bash
make ps
```

This stage extracts document content, applies cleanup/processors, and can enrich pages with table/image captioning before chunking.

### 3) Chunk the parsed documents

```bash
python chunk_processor.py
```

or:

```bash
make cs
```

This creates vector-ready chunk records for each configured embedding model. It is the bridge between extracted document structure and retrieval indexing. To attach to the running chunking tmux session, use:

```bash
make ccs
```

### 4) Build the QA index in Qdrant

```bash
python qa_indexer.py
```

or:

```bash
make iqa
```

This script loads QA records, creates embeddings, caches them, and indexes them as a searchable collection in Qdrant.

### 5) Build the source document index

```bash
python src_indexer.py
```

or:

```bash
make is
```

This indexes source chunks into Qdrant using the configured embedder and chunking method.

## Generating experiments

The main experiment entry point is `generate.py`.

```bash
python generate.py
```

or:

```bash
make gen
```

This script creates a matrix of experiment setups combining:

- several embedding models
- multiple document modalities
- QA sample subsets
- retrieval generation runs

Outputs are saved in timestamped folders under `outputs/`.

## Evaluating runs

```bash
python evaluate.py
```

or:

```bash
make eval
```

This reads the generated outputs, computes benchmark metrics, and writes evaluation summaries to the run directory.

The evaluation combines:

- BLEU
- ROUGE-1 / ROUGE-2 / ROUGE-L
- context precision
- context recall
- faithfulness
- answer relevancy
- semantic similarity

It also performs statistical comparison across experiment runs.

## Output artifacts

Each run writes JSON and Markdown summaries under a timestamped directory, for example:

```text
outputs/
└── run-20260922_094130/
    ├── generated/
    │   └── ...
    ├── evaluated/
    │   ├── experiments_benchmark_details.json
    │   └── ...
    └── statistically_evaluated/
        ├── multi_statistical_summary_details.json
        ├── pair_statistical_summary_details.json
        ├── multi_statistical_summary_readable.md
        └── pair_statistical_summary_readable.md
```

## Main entry points

### `pdf_processor.py`

Runs the document extraction, cleanup, enrichment, and structured node-building pipeline.

### `chunk_processor.py`

Takes parsed documents and turns them into semantic chunk records ready for vector indexing.

This stage is also exposed by the shortcuts:

```bash
make cs
make ccs
```

### `qa_parser.py`
Normalizes raw QA JSON and creates deterministic QA IDs for retrieval and evaluation.

### `qa_indexer.py`

Indexes the QA dataset into Qdrant using per-embedder caches.

### `src_indexer.py`

Indexes source document chunks into Qdrant.

### `generate.py`

Defines the experiment matrix and runs the RAG evaluation generation loop.

### `evaluate.py`

Loads generated experiment results and computes benchmark statistics.

### `main.py`

A general evaluation entry point for experiment execution and benchmarking.

## Makefile shortcuts

Common commands:

```bash
make qdrant
make ollama
make pqa
make ps
make cs
make iqa
make is
make gen
make eval
make kill
```

Tmux-attached views:

```bash
make cps
make ccs
make ciqa
make cis
make cgen
make ceval
```

## Recommended first run

A typical starting workflow is:

```bash
export PROJECT_ROOT="/path/to/MNT-MRAG-Refactored"
make qdrant
make ollama
make pqa
make ps
make cs
make iqa
make is
python generate.py
python evaluate.py
```

This prepares the dataset, builds the vector stores, generates answers, and creates benchmark summaries.

## Practical notes for researchers

- This repo is designed for repeated experiments and benchmarking, not as a hosted application.
- `PROJECT_ROOT` must be set correctly; otherwise the storage layout and output directories will be wrong.
- Qdrant and Ollama need to be running before generation or evaluation begins.
- The project uses dynamic collection naming based on embedder and chunking configuration, so multiple indexed configurations can coexist.
- Embedding caches are written to disk to reduce redundant compute when re-running the same experiments.

## Execution summary

At a high level, the repository follows this flow:

`raw PDF + QA dataset -> parsed documents -> semantic chunks -> Qdrant indexes -> retrieval -> generation -> evaluation`

That is the core research workflow this project is built around.
