from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Self
import json


@dataclass(slots=True)
class BaseConfig:
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

    def save_json(self, path: str | Path) -> Path:

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.to_dict(),
                f,
                indent=4,
                sort_keys=True,
                ensure_ascii=False
            )

        return path

    @classmethod
    def load_json(cls, path: str | Path) -> Self:

        path = Path(path)

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return cls.from_dict(data)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        pass

class BaseMLConfig(BaseConfig):
    random_state: int | None = None
    device: str = "cpu"

@dataclass(slots=True)
class BasePretrainedConfig(BaseMLConfig):

    model_name: str | None = None

    revision: str | None = None

    cache_dir: str | Path | None = None

    local_files_only: bool = False

    trust_remote_code: bool = False

    dtype: str | None = None

    def validate(self) -> None:

        if self.model_name is None:
            raise ValueError(
                "model_name cannot be empty."
            )

@dataclass
class BaseCaptioningConfig(BasePretrainedConfig):

    max_new_tokens: int = 256

    temperature: float = 0.7

    top_p: float = 0.9

    do_sample: bool = True

    def validate(self) -> None:

        super().validate()

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be > 0."
            )

        if not 0 <= self.temperature:
            raise ValueError(
                "temperature must be >= 0."
            )

        if not 0 <= self.top_p <= 1:
            raise ValueError(
                "top_p must be between 0 and 1."
            )

@dataclass
class OllamaCaptioningConfig(BaseCaptioningConfig):
    host: str = "http://localhost:11434"

@dataclass
class BaseTextGeneratorConfig(BasePretrainedConfig):

    max_new_tokens: int = 512

    temperature: float = 0.7

    top_p: float = 0.9

    do_sample: bool = True

    def validate(self) -> None:

        super().validate()

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be > 0."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

        if not 0 <= self.top_p <= 1:
            raise ValueError(
                "top_p must be between 0 and 1."
            )

class ImageExtractorConfig(BaseConfig):
    ...

class TableExtractorConfig(BaseConfig):
    ...