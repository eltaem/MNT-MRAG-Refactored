from __future__ import annotations

from dataclasses import dataclass

from langchain_core.outputs import (
    Generation,
    LLMResult,
)
from langchain_core.prompt_values import PromptValue

from ragas.llms.base import BaseRagasLLM

from src.models import BaseTextGeneratorModel


@dataclass
class RAGASLLMAdapter(BaseRagasLLM):

    generator: BaseTextGeneratorModel

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:

        prompt_text = prompt.to_string()

        texts = self.generator.generate_many(
            prompt=prompt_text,
            n=n,
            temperature=temperature,
            stop=stop,
        )

        return LLMResult(
            generations=[
                [
                    Generation(text=text)
                    for text in texts
                ]
            ]
        )

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:

        prompt_text = prompt.to_string()

        texts = await self.generator.agenerate_many(
            prompt=prompt_text,
            n=n,
            temperature=temperature,
            stop=stop,
        )

        return LLMResult(
            generations=[
                [
                    Generation(text=text)
                    for text in texts
                ]
            ]
        )

    def is_finished(
        self,
        response: LLMResult,
    ) -> bool:

        return True