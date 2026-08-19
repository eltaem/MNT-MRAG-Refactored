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

        schema = response_model.model_json_schema()

        required_fields = ", ".join(
            schema.get("required", [])
        )

        return self.generator.generate_structured(
            prompt=prompt,
            response_model=response_model,
            system_prompt=self.generator.config.system_prompt.format(
                required_fields=required_fields,
                language=self.generator.config.language,
            ),
        )

    async def agenerate(
        self,
        prompt: str,
        response_model: Type[InstructorTypeVar],
    ) -> InstructorTypeVar:

        schema = response_model.model_json_schema()

        required_fields = ", ".join(
            schema.get("required", [])
        )

        return await self.generator.agenerate_structured(
            prompt=prompt,
            response_model=response_model,
            system_prompt=self.generator.config.system_prompt.format(
                required_fields=required_fields,
                language=self.generator.config.language,
            ),
        )