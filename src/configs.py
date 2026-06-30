from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from enum import StrEnum
from src.utility import LogLevel

class TextMergeMode(StrEnum):
    PAGE = "page"
    DOCUMENT = "document"

_PENDAHULUAN_PATTERNS: list[str] = [
    "BAB I\nPENDAHULUAN\n",
    "BAB 1\nPENDAHULUAN\n",
    "BABI\nPENDAHULUAN\n",
    "BAB I. PENDAHULUAN\n",
    "BAB 1. PENDAHULUAN\n",
    "BAB I\n",
    "BAB 1\nPENDAHULUAN",
    "BAB 1\n",
    "PENDAHULUAN\n",
    "A. IDENTITAS PASIEN\n",
    "SKRINING GIZI\n",
    "DATA DASAR\n",
]

DEFAULT_FRONT_MATTER_PATTERNS: list[str] = [
    "BAB I PENDAHULUAN",
    "BAB 1 PENDAHULUAN",
    "PENDAHULUAN",
    "A IDENTITAS PASIEN",
    "SKRINING GIZI",
    "DATA DASAR",
]

class BaseConfig(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str | None = None

    def save_json(
        self,
        path: str | Path,
    ) -> Path:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            self.model_dump_json(indent=4),
            encoding="utf-8",
        )

        return path

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ):

        path = Path(path)

        return cls.model_validate_json(
            path.read_text(
                encoding="utf-8"
            )
        )

    log_level: LogLevel = LogLevel.INFO

# ============================================================
# ML
# ============================================================

class BaseMLConfig(BaseConfig):

    random_state: int | None = None

    device: str = "cpu"


# ============================================================
# BACKENDS
# ============================================================

class BackendConfig(BaseConfig):
    pass


class OllamaBackendConfig(BackendConfig):

    host: str = "http://localhost:11434"

    timeout: int = Field(
        default=600,
        gt=0,
    )

    keep_alive: str | int | None = None

    think: bool = False


class TransformerBackendConfig(
    BackendConfig
):

    revision: str | None = None

    cache_dir: str | Path | None = None

    local_files_only: bool = False

    trust_remote_code: bool = False

    dtype: str | None = None


# ============================================================
# PRETRAINED
# ============================================================

class BasePretrainedConfig(BaseMLConfig):

    model_name: str


# ============================================================
# GENERATION PARAMETERS
# ============================================================

class TextGenerationConfig(BasePretrainedConfig):

    max_new_tokens: int = Field(
        default=256,
        gt=0,
    )

    temperature: float = Field(
        default=0.7,
        ge=0,
    )

    top_p: float = Field(
        default=0.9,
        ge=0,
        le=1,
    )

    do_sample: bool = True


# ============================================================
# CAPTIONING
# ============================================================

class BaseCaptioningConfig(TextGenerationConfig):
    language: str = Field(
        default="indonesian",
        description="Language used for generated captions.",
    )

class BaseImageCaptioningConfig(BaseCaptioningConfig):

    system_prompt: str = (
        "You are an expert image captioning assistant. "
        "Produce concise, factual, and objective descriptions "
        "in {language}. Describe only information that is "
        "clearly visible in the image. Do not speculate, "
        "infer hidden information, or invent details that "
        "cannot be directly observed."
    )

    user_prompt: str = (
        "Describe this image in one concise paragraph. "
        "Include only the most important visible information."
    )

class BaseTableCaptioningConfig(BaseCaptioningConfig):

    system_prompt: str = (
        "You are an expert table summarization assistant. "
        "Produce concise, factual, and objective summaries "
        "in {language}. Describe only information explicitly "
        "present in the table. Do not speculate, infer missing "
        "values, infer diagnoses, infer causal relationships, "
        "or invent conclusions that are not directly supported "
        "by the table."
    )

    user_prompt: str = (
        "Describe the following table in one concise paragraph. "
        "Summarize the most important information, key values, "
        "and any explicit patterns shown in the table."
    )

class OllamaImageCaptioningConfig(BaseImageCaptioningConfig):

    backend: OllamaBackendConfig = Field(
        default_factory=OllamaBackendConfig
    )


class OllamaTableCaptioningConfig(BaseTableCaptioningConfig):

    backend: OllamaBackendConfig = Field(
        default_factory=OllamaBackendConfig
    )


class TransformerImageCaptioningConfig(BaseImageCaptioningConfig):

    backend: TransformerBackendConfig = Field(
        default_factory=TransformerBackendConfig
    )


class TransformerTableCaptioningConfig(BaseTableCaptioningConfig):

    backend: TransformerBackendConfig = Field(
        default_factory=TransformerBackendConfig
    )


# ============================================================
# TEXT GENERATION
# ============================================================

class BaseTextGeneratorConfig(TextGenerationConfig):
    pass


class OllamaTextGeneratorConfig(BaseTextGeneratorConfig):

    backend: OllamaBackendConfig = Field(
        default_factory=OllamaBackendConfig
    )

class TransformerTextGeneratorConfig(BaseTextGeneratorConfig):

    backend: TransformerBackendConfig = Field(
        default_factory=TransformerBackendConfig
    )

# ============================================================
# EXTRACTORS
# ============================================================

class BaseExtractorConfig(BaseConfig):
    pass

class TextExtractorConfig(BaseExtractorConfig):

    merge_spans: bool = True

    strip_whitespace: bool = True

    skip_empty_blocks: bool = True

    min_text_length: int = Field(
        default=1,
        ge=0,
    )

class ImageExtractorConfig(BaseExtractorConfig):

    image_dir: Path = Path("data/images")

    min_width: int = Field(
        default=32,
        gt=0,
    )

    min_height: int = Field(
        default=32,
        gt=0,
    )

    deduplicate: bool = True

class TableExtractorConfig(BaseExtractorConfig):

    skip_empty_tables: bool = True

    min_rows: int = Field(
        default=1,
        ge=1,
    )

    min_columns: int = Field(
        default=1,
        ge=1,
    )

class PDFExtractorConfig(BaseConfig):

    text: TextExtractorConfig = Field(
        default_factory=TextExtractorConfig
    )

    image: ImageExtractorConfig = Field(
        default_factory=ImageExtractorConfig
    )

    table: TableExtractorConfig = Field(
        default_factory=TableExtractorConfig
    )

# ============================================================
# PROCESSORS
# ============================================================

class BaseNodeProcessorConfig(
    BaseConfig
):
    """
    Base configuration for processors that
    structurally transform nodes.

    Examples:
    - RemoveFrontMatterProcessor
    - MergeTextProcessor
    - SortNodeProcessor
    """

    enabled: bool = True

class RemoveBeforePatternProcessorConfig(
    BaseNodeProcessorConfig,
):
    """
    Remove every node before the first text node
    matching one of the configured patterns.
    """

    patterns: list[str] = Field(
        default_factory=list,
    )

    ignore_case: bool = True

    normalize_whitespace: bool = True

    normalize_punctuation: bool = True

    startswith: bool = True

class BaseNodeEnricherConfig(
    BaseConfig
):
    """
    Base configuration for enrichers that
    add information to nodes.

    Examples:
    - ImageCaptionEnricher
    - TableCaptionEnricher
    - OCREnricher
    """

    enabled: bool = True

    batch_size: int = Field(
        default=8,
        gt=0,
    )

    skip_existing: bool = Field(
        default=True,
    )

class ImageCaptionEnricherConfig(
    BaseNodeEnricherConfig
):
    pass


class TableCaptionEnricherConfig(
    BaseNodeEnricherConfig
):
    pass

class TextMergerConfig(BaseNodeProcessorConfig,):
    """
    Merge consecutive TextNodes.

    PAGE:
        Do not merge across pages.

    DOCUMENT:
        Allow merging across pages.

    ImageNode and TableNode always
    create hard boundaries.
    """

    mode: TextMergeMode = TextMergeMode.PAGE

    separator: str = "\n"

    merge_metadata: bool = False

class ChunkBuilderConfig(BaseConfig):
    pass
class EmbedderConfig(BaseConfig):
    pass
class VectorStoreConfig(BaseConfig):
    pass

class PDFProcessorConfig(BaseConfig):
    pdf_concurrency: int = Field(
        default=2,
        gt=0,
    )