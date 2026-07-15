from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypeVar
import json
import logging
import numpy as np
from pydantic import BaseModel
from transformers import pipeline

from src.utility import clean_text, serialize_table
from src.configs import BaseConfig, BaseImageCaptioningConfig, BasePretrainedConfig, BaseCaptioningConfig, BaseTableCaptioningConfig, BaseTextGeneratorConfig, OllamaImageCaptioningConfig, OllamaTableCaptioningConfig, OllamaTextGeneratorConfig, TransformerImageCaptioningConfig
InstructorTypeVar = TypeVar("T", bound=BaseModel)
import ollama

class MyBaseModel(ABC):

    def __init__(self, config: BaseConfig):
        self.config = config

        self.name = (config.name if config.name is not None else self.__class__.__name__)

        self.logger = logging.getLogger(self.name)

    def to_dict(self) -> dict:
        return {
            "model_type": self.__class__.__name__,
            "config": self.config.to_dict(),
        }

    def to_json(self, path: str | Path) -> None:

        def json_converter(obj):

            if isinstance(obj, (np.ndarray, list, tuple)):
                return np.array(obj).tolist()

            return str(obj)

        with open(path, "w", encoding="utf-8") as f:

            json.dump(self.to_dict(), f, indent=4, default=json_converter)

    def to_folder(self, directory: str | Path) -> Path:

        directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)

        self.config.to_json(directory / "config.json")

        self.logger.info(f"Successfully saved model to {directory}")

        return directory

    # @classmethod
    # @abstractmethod
    # def from_folder(cls, directory: str | Path) -> Self:
    #     pass

class BasePretrainedModel(MyBaseModel):

    def __init__(self, config: BasePretrainedConfig,
    ):
        super().__init__(config)

        self.is_loaded: bool = False

    @abstractmethod
    def load_pretrained(self) -> None:
        """
        Load model weights and related artifacts.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    def _check_is_loaded(self) -> None:

        if not self.is_loaded:
            raise RuntimeError(
                f"Model '{self.name}' has not been loaded."
            )

    def to_dict(self) -> dict:

        data = super().to_dict()

        data["runtime"] = {
            "is_loaded": self.is_loaded
        }

        return data

class BaseCaptioningModel(BasePretrainedModel):

    def __init__(self, config: BaseCaptioningConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def caption(self, input_data: Any,**kwargs,) -> str:
        """
        Generate a caption from the supplied input.
        """
        pass

class BaseTextGeneratorModel(BasePretrainedModel):

    def __init__(
        self,
        config: BaseTextGeneratorConfig,
    ):
        super().__init__(config)

        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a single completion.
        """
        ...

    def generate_many(
        self,
        prompt: str,
        n: int = 1,
        *,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate multiple completions.

        Concrete implementations may override this
        with native batched generation.
        """

        return [
            self.generate(
                prompt=prompt,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )
            for _ in range(n)
        ]

    async def agenerate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Asynchronous single completion.

        Concrete implementations may override this
        with native async inference.
        """

        return self.generate(
            prompt=prompt,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )

    async def agenerate_many(
        self,
        prompt: str,
        n: int = 1,
        *,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Asynchronous multiple completions.

        Concrete implementations may override this
        with native async batched inference.
        """

        return [
            await self.agenerate(
                prompt=prompt,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )
            for _ in range(n)
        ]
    
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
        **kwargs,
    ) -> InstructorTypeVar:
        ...

    @abstractmethod
    async def agenerate_structured(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
        **kwargs,
    ) -> InstructorTypeVar:
        ...

class BaseImageCaptioningModel(BaseCaptioningModel):
    def __init__(
        self,
        config: BaseImageCaptioningConfig,
    ):
        super().__init__(config)
        self.config = config

class BaseTableCaptioningModel(BaseCaptioningModel):
    def __init__(
        self,
        config: BaseTableCaptioningConfig,
    ):
        super().__init__(config)
        self.config = config

class TransformerImageCaptioningModel(BaseImageCaptioningModel):

    def __init__(self, config: TransformerImageCaptioningConfig):
        super().__init__(config)
        self.config = config

        self._pipe = None

    def load_pretrained(self) -> None:
        self._pipe = pipeline(
            "image-text-to-text",
            model=self.config.model_name,
            device=self.config.device,
            trust_remote_code=self.config.backend.trust_remote_code
            )
        self.is_loaded = True

    def unload(self) -> None:
        self._pipe = None
        self.is_loaded = False

    def caption(self, input_data: str | Path, prompt: str | None = None, **kwargs,) -> str:
        self._check_is_loaded()

        image_path = str(input_data)

        if prompt is None:
            prompt = self.config.user_prompt

        messages = [
            {
                "role": "system",
                "content": (
                    self.config.system_prompt
                    .format(
                        language=self.config.language
                    )
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        result = self._pipe(
            text=messages,
        )

        generated_text = (
            result[0]
            ["generated_text"][1]
            ["content"]
        )

        return clean_text(
            generated_text
        )
    
class OllamaImageCaptioningModel(BaseImageCaptioningModel):
    def __init__(
        self,
        config: OllamaImageCaptioningConfig,
    ):
        super().__init__(config)
        self.config = config
        self._client: ollama.Client | None = None

    def load_pretrained(self) -> None:
        self._client = ollama.Client(
            host=self.config.backend.host
        )
        self.is_loaded = True

    def unload(self) -> None:
        self._client = None
        self.is_loaded = False

    def caption(self, input_data: str | Path, prompt: str | None = None, **kwargs,) -> str:

        self._check_is_loaded()

        image_path = str(input_data)

        if prompt is None:
            prompt = self.config.user_prompt

        system_prompt = (
            self.config.system_prompt
            .format(
                language=self.config.language
            )
        )

        response = self._client.chat(
            model=self.config.model_name,
            think=self.config.backend.think,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                },
            ],
            options={
                "temperature":
                    self.config.temperature,
                "top_p":
                    self.config.top_p,
                "num_predict":
                    self.config.max_new_tokens,
            },
        )

        return clean_text(
            response.message.content
        )

class OllamaTableCaptioningModel(BaseTableCaptioningModel):
    def __init__(self, config: OllamaTableCaptioningConfig):
        super().__init__(config)
        self.config = config

        self._client = None

    def load_pretrained(self) -> None:
        self._client = ollama.Client(host=self.config.backend.host)
        self.is_loaded = True

    def unload(self) -> None:
        self._client = None
        self.is_loaded = False

    def caption(
        self,
        input_data: Any,
        prompt: str | None = None,
        **kwargs,
    ) -> str:

        self._check_is_loaded()

        if prompt is None:
            prompt = self.config.user_prompt

        system_prompt = (
            self.config.system_prompt
            .format(
                language=self.config.language
            )
        )

        table_text = serialize_table(
            input_data
        )

        response = self._client.chat(
            model=self.config.model_name,
            think=self.config.backend.think,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"TABLE:\n{table_text}"
                    ),
                },
            ],
            options={
                "temperature":
                    self.config.temperature,
                "top_p":
                    self.config.top_p,
                "num_predict":
                    self.config.max_new_tokens,
            },
        )

        return clean_text(
            response.message.content
        )

    @classmethod
    def from_folder(cls, directory: str | Path,) -> "OllamaTableCaptioningModel":

        config = OllamaTableCaptioningConfig.from_json(
            Path(directory) / "config.json"
        )

        model = cls(config)

        model.load_pretrained()

        return model

class OllamaTextGenerator(
    BaseTextGeneratorModel,
):

    def __init__(
        self,
        config: OllamaTextGeneratorConfig,
    ):
        super().__init__(config)

        self.config = config

        self._client: ollama.Client | None = None

        self._aclient: ollama.AsyncClient | None = None

    def load_pretrained(
        self,
    ) -> None:

        self._client = ollama.Client(
            host=self.config.backend.host,
        )

        self._aclient = ollama.AsyncClient(
            host=self.config.backend.host,
        )

        self.is_loaded = True

    def unload(
        self,
    ) -> None:

        self._client = None

        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:

        self._check_is_loaded()

        if system_prompt is None:
            system_prompt = (
                self.config.system_prompt.format(
                    language=self.config.language,
                )
            )

        response = self._client.chat(
            model=self.config.model_name,
            think=self.config.backend.think,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": kwargs.get(
                    "temperature",
                    self.config.temperature,
                ),
                "top_p": kwargs.get(
                    "top_p",
                    self.config.top_p,
                ),
                "num_predict": kwargs.get(
                    "max_new_tokens",
                    self.config.max_new_tokens,
                ),
            },
        )

        return clean_text(
            response.message.content,
        )

    @classmethod
    def from_folder(
        cls,
        directory: str | Path,
    ) -> "OllamaTextGenerator":

        config = OllamaTextGeneratorConfig.from_json(
            Path(directory) / "config.json",
        )

        model = cls(config)

        model.load_pretrained()

        return model
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
        system_prompt: str | None = None,
        **kwargs,
    ) -> BaseModel:

        self._check_is_loaded()

        if system_prompt is None:
            system_prompt = (
                self.config.system_prompt.format(
                    language=self.config.language,
                )
            )

        response:ollama.ChatResponse = self._client.chat(
            model=self.config.model_name,
            think=self.config.backend.think,
            format=response_model.model_json_schema(),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": kwargs.get(
                    "temperature",
                    self.config.temperature,
                ),
                "top_p": kwargs.get(
                    "top_p",
                    self.config.top_p,
                ),
                "num_predict": kwargs.get(
                    "max_new_tokens",
                    self.config.max_new_tokens,
                ),
            },
        )

        return response_model.model_validate_json(
            response.message.content,
        )
    
    async def agenerate_structured(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
        system_prompt: str | None = None,
        **kwargs,
    ) -> BaseModel:

        self._check_is_loaded()

        if system_prompt is None:
            system_prompt = (
                self.config.system_prompt.format(
                    language=self.config.language,
                )
            )

        response:ollama.ChatResponse = await self._aclient.chat(
            model=self.config.model_name,
            think=self.config.backend.think,
            format=response_model.model_json_schema(),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": kwargs.get(
                    "temperature",
                    self.config.temperature,
                ),
                "top_p": kwargs.get(
                    "top_p",
                    self.config.top_p,
                ),
                "num_predict": kwargs.get(
                    "max_new_tokens",
                    self.config.max_new_tokens,
                ),
            },
        )

        return response_model.model_validate_json(
            response.message.content,
        )