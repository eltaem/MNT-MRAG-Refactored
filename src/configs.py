from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

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
    pass


class BaseImageCaptioningConfig(BaseCaptioningConfig):
    pass


class BaseTableCaptioningConfig(BaseCaptioningConfig):
    pass


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

    image_dir: Path = Path("images")

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

class PDFExtractorConfig(BaseExtractorConfig):
    merge_pages:bool = False
    pass