from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM

from src.models import BaseTextGeneratorModel

InstructorTypeVar = TypeVar("T", bound=BaseModel)

@dataclass
class RAGASLLMAdapter(
    InstructorBaseRagasLLM,
):

    generator: BaseTextGeneratorModel

    def generate(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
    ) -> InstructorTypeVar:

        return self.generator.generate_structured(
            prompt=prompt,
            response_model=response_model,
        )

    async def agenerate(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
    ) -> InstructorTypeVar:

        return await self.generator.agenerate_structured(
            prompt=prompt,
            response_model=response_model,
        )