from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Generic, TypeVar

@dataclass
class EvaluationResult(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        pass
    
    @abstractmethod
    def to_markdown(self) -> str:
        """
        Convert evaluation results into a markdown report.
        """
        pass

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "metrics.json", "w") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,
                default=str,
            )
        
        with open(
            directory / "report.md", "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

        return directory

ResultT = TypeVar("ResultT")

class Evaluator(ABC, Generic[ResultT]):
    """
    Abstract base class for model evaluators.
    """

    @abstractmethod
    def compute_metrics(self, *args, **kwargs) -> dict:
        """
        Compute evaluation metrics.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> ResultT:
        """
        Perform evaluation and return an EvaluationResult.
        """
        pass