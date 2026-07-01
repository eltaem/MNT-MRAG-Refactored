import json
from typing import Any
import uuid

import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, computed_field
import pandas as pd
from src.utility import (
    load_json,
    load_models,
    save_models,
)

load_dotenv()

def generate_qa_id(
    qa: "QAItem",
) -> str:
    """
    Generate a deterministic UUID from QA content.

    The same question-answer pair will always produce
    the same UUID across runs.
    """

    payload: dict[str, Any] = {
        "question": qa.question,
        "answer": qa.answer,
    }

    payload_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            payload_json,
        )
    )

class QAItem(BaseModel):

    question: str

    answer: str

    @computed_field
    @property
    def qa_id(self) -> str:
        return generate_qa_id(self)

class QAEmbedding(BaseModel):

    qa_id: str

    question_embedding: list[float]

    answer_embedding: list[float]

class QARecord(BaseModel):

    qa: QAItem

    embedding: QAEmbedding

    @classmethod
    def from_embeddings(
        cls,
        qa_items: list[QAItem],
        qa_embeddings: list[QAEmbedding],
    ) -> list["QARecord"]:

        embedding_map = {
            emb.qa_id: emb
            for emb in qa_embeddings
        }

        missing = [
            qa.qa_id
            for qa in qa_items
            if qa.qa_id not in embedding_map
        ]

        if missing:
            raise ValueError(
                f"Missing embeddings for {len(missing)} QA items."
            )

        return [
            cls(
                qa=qa,
                embedding=embedding_map[qa.qa_id],
            )
            for qa in qa_items
        ]

class QAEmbeddingParser:

    @staticmethod
    def from_json(
        path: str | Path,
    ) -> list[QAEmbedding]:

        return load_models(
            path=path,
            model_cls=QAEmbedding,
        )

    @staticmethod
    def to_json(
        embeddings: list[QAEmbedding],
        path: str | Path,
    ) -> Path:

        return save_models(
            models=embeddings,
            path=path,
        )

class QAParser:

    @staticmethod
    def from_raw_json(
        path: str | Path,
    ) -> list[QAItem]:

        payload = load_json(path)

        qa_items: list[QAItem] = []

        for row in payload:

            anthropometric = (
                row.get(
                    "anthropometric_assessment",
                    "",
                )
                .strip()
            )
            anthropometric = " ".join(
                anthropometric.split()
            )

            diagnosis = (
                row.get(
                    "medical_diagnosis",
                    "",
                )
                .strip()
            )
            diagnosis = " ".join(
                diagnosis.split()
            )

            question = (
                row.get(
                    "question",
                    "",
                )
                .strip()
            )
            question = " ".join(
                question.split()
            )

            final_question = (
                f"{anthropometric} | "
                f"{diagnosis} | "
                f"{question}"
            )

            answer = (
                row.get(
                    "answer",
                    {},
                )
                .get(
                    "text",
                    [],
                )
            )

            if isinstance(answer, list):
                answer = " ".join(
                    a.strip()
                    for a in answer
                )

            answer = " ".join(
                str(answer).split()
            )

            qa_items.append(
                QAItem(
                    question=final_question,
                    answer=answer,
                )
            )

        return qa_items

    @staticmethod
    def to_json(
        qa_items: list[QAItem],
        path: str | Path,
    ) -> Path:

        return save_models(
            qa_items,
            path,
        )

    @staticmethod
    def from_json(
        path: str | Path,
    ) -> list[QAItem]:

        return load_models(
            path,
            QAItem,
        )
    
    @staticmethod
    def to_csv(
        qa_items: list[QAItem],
        path: str | Path,
    ) -> Path:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df = pd.DataFrame(
            [
                {
                    "question": qa.question,
                    "answer": qa.answer,
                }
                for qa in qa_items
            ]
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        return path

def main():
    from src.configs import STORAGE_CONFIG

    raw_json_path = STORAGE_CONFIG.qa_raw_json

    parsed_json_path = STORAGE_CONFIG.qa_json

    parsed_csv_path = STORAGE_CONFIG.qa_csv

    qa_items = QAParser.from_raw_json(
        raw_json_path
    )

    print(
        f"Total QA: {len(qa_items)}"
    )

    if parsed_json_path.exists():
        print(
            "Parsed QA via JSON already exists at",
            parsed_json_path.relative_to(
                Path.cwd()
            ),
        )
    else:
        QAParser.to_json(
            qa_items,
            parsed_json_path,
        )
        print(
            "Saved parsed QA via JSON to",
            parsed_json_path.relative_to(
                Path.cwd()
            ),
        )
    
    if parsed_csv_path.exists():
        print(
            "Parsed QA via csv already exists at",
            parsed_csv_path.relative_to(
                Path.cwd()
            )
        )
    else:
        QAParser.to_csv(
                qa_items,
                parsed_csv_path,
            )
        print(
            "Saved parsed QA via csv to",
            parsed_csv_path.relative_to(
                Path.cwd()
            ),
        )

if __name__ == "__main__":
    main()