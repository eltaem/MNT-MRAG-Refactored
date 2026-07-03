from __future__ import annotations
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, computed_field
import pandas as pd
from src.record import BaseRecord
from src.utility import (
    deterministic_uuid,
    load_json,
    load_models,
    save_models,
)
from src.vector_store import SearchResult

load_dotenv()

def generate_qa_id(qa: QAItem) -> str:
    return deterministic_uuid({
        "question": qa.question,
        "answer": qa.answer,
    })

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

class QARecord(BaseRecord):

    qa: QAItem

    embedding: QAEmbedding

    @classmethod
    def from_embeddings(
        cls,
        qa_items: list[QAItem],
        qa_embeddings: list[QAEmbedding],
    ) -> list[QARecord]:

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

    def to_payload(self) -> dict:
        return {
            "qa": self.qa.model_dump(
                mode="json",
            ),
            "answer_embedding": self.embedding.answer_embedding,
        }

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
    ) -> "QARecord":

        if result.embedding is None:
            raise ValueError(
                "SearchResult does not contain vectors."
            )

        return cls(
            qa=QAItem.model_validate(
                result.payload["qa"],
            ),
            embedding=QAEmbedding(
                qa_id=result.id,
                question_embedding=result.embedding,
                answer_embedding=result.payload[
                    "answer_embedding"
                ],
            ),
        )

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

def parse_qa():
    from src.configs import STORAGE_CONFIG

    qa_items = QAParser.from_raw_json(
        STORAGE_CONFIG.qa_raw_json
    )

    print(
        f"Total QA: {len(qa_items)}"
    )

    if STORAGE_CONFIG.qa_json.exists():
        print(
            "Parsed QA via JSON already exists at",
            STORAGE_CONFIG.qa_json.relative_to(
                Path.cwd()
            ),
        )
    else:
        QAParser.to_json(
            qa_items,
            STORAGE_CONFIG.qa_json,
        )
        print(
            "Saved parsed QA via JSON to",
            STORAGE_CONFIG.qa_json.relative_to(
                Path.cwd()
            ),
        )
    
    if STORAGE_CONFIG.qa_csv.exists():
        print(
            "Parsed QA via csv already exists at",
            STORAGE_CONFIG.qa_csv.relative_to(
                Path.cwd()
            )
        )
    else:
        QAParser.to_csv(
                qa_items,
                STORAGE_CONFIG.qa_csv,
            )
        print(
            "Saved parsed QA via csv to",
            STORAGE_CONFIG.qa_csv.relative_to(
                Path.cwd()
            ),
        )

if __name__ == "__main__":
    parse_qa()