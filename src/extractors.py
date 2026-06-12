from __future__ import annotations
from pathlib import Path

class ImageExtractor:
    ...

class TableExtractor:
    ...
class PDFExtractor:
    def __init__(self):
        ...

    @classmethod
    def from_config(cls) -> "PDFExtractor":
        ...

    def extract_pdf(self, path:str | Path) -> Path:
        ...

    async def aextract_pdf(self, path:str | Path) -> Path:
        ...

    def extract_pdfs(self, dir:str | Path) -> Path:
        ...

    async def aextract_pdfs(self, dir:str | Path) -> Path:
        ...