import asyncio
import torch

from pydantic import Field, PrivateAttr
from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from ragas.embeddings.base import BaseRagasEmbedding
from llama_index.core.embeddings import BaseEmbedding
from src.configs import EmbedderConfig
from src.utility import clear_gpu

class Embedder(Embeddings, BaseRagasEmbedding, BaseEmbedding):
    """
    Universal embedding wrapper for:
    - SentenceTransformers
    - LangChain
    - Ragas
    - LlamaIndex

    Features:
    ----------
    - target_dimension truncation
    - fp16 inference
    - VRAM optimized
    - embedding cache
    - async support
    """

    modele: SentenceTransformer | None = Field(default=None,exclude=True)

    config: EmbedderConfig = Field(default_factory=EmbedderConfig)

    _full_dimension: int = PrivateAttr(default=-1)

    _cache: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # =========================================================
    # INIT
    # =========================================================

    def __init__(
        self,
        config: EmbedderConfig,
        **kwargs,
    ):

        object.__setattr__(self, "modele", None)

        object.__setattr__(self, "config", config)

        object.__setattr__(self, "_cache", {})

        object.__setattr__(self, "_full_dimension", -1)
        
        clear_gpu()

    # =========================================================
    # STRING
    # =========================================================
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def embedding_dimension(self):

        if self.config.target_dimension is not None:
            return self.config.target_dimension

        if self._full_dimension <= 0:
            raise RuntimeError(
                "Embedding dimension unknown. "
                "Call load() first or specify target_dimension."
            )

        return self._full_dimension

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def load(self):

        if self.modele is not None:
            return

        clear_gpu()

        model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            trust_remote_code=True,
            model_kwargs={
                "dtype": self.config.dtype,
            },
        )

        model.max_seq_length = self.config.max_seq_length
        model.eval()

        object.__setattr__(self, "modele", model,)

        sample = model.encode("-")
        sample = sample.tolist()
        object.__setattr__(self, "_full_dimension", len(sample),)

        if (self.config.target_dimension is not None) and (self.config.target_dimension > self._full_dimension):
                raise ValueError(
                    f"target_dimension={self.config.target_dimension} "
                    f"exceeds model dimension={self._full_dimension}"
                )

    def _truncate_embedding(self, emb: list[float]) -> list[float]:

        if self.config.target_dimension is None:
            return emb

        return emb[:self.config.target_dimension]

    def _encode_single(self, text: str) -> list[float]:
        self._require_model()
        with torch.inference_mode():

            emb = self.modele.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=1,
            )

        emb = emb.tolist()

        emb = self._truncate_embedding(emb)

        return emb

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        self._require_model()
        if not texts:
            return []

        with torch.inference_mode():

            embeddings = self.modele.encode(
                texts,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                convert_to_tensor=False,
                show_progress_bar=len(texts) > 32,
                batch_size=self.config.batch_size,
            )

        embeddings = embeddings.tolist()

        if self.config.target_dimension is not None:
            embeddings = [self._truncate_embedding(emb) for emb in embeddings]

        return embeddings

    # =========================================================
    # LANGCHAIN
    # =========================================================

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        uncached = [t for t in texts if t not in self._cache]

        if uncached:

            new_embeddings = self._encode_batch(uncached)

            for text, emb in zip(uncached, new_embeddings):
                self._cache[text] = emb

        return [self._cache[t] for t in texts]

    # =========================================================
    # RAGAS
    # =========================================================

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            None,
            self.embed_query,
            text
        )
    
    async def aembed_query(self, text: str) -> list[float]:
        return await self._aget_embedding(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(None, self.embed_documents, texts)

    # =========================================================
    # LLAMAINDEX
    # =========================================================

    def _get_query_embedding(self, text: str) -> list[float]:
        return self.embed_query(text)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def _aget_query_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    # =========================================================
    # CACHE
    # =========================================================

    def _get_embedding(self, text: str) -> list[float]:

        if text in self._cache:
            return self._cache[text]

        emb = self._encode_single(text)

        self._cache[text] = emb

        return emb

    async def _aget_embedding(self, text: str) -> list[float]:

        if text in self._cache:
            return self._cache[text]

        loop = asyncio.get_event_loop()

        emb = await loop.run_in_executor(None, self._encode_single, text)

        self._cache[text] = emb

        return emb

    def clear_cache(self):

        self._cache.clear()

    @property
    def cache_size(self) -> int:

        return len(self._cache)
    
    @property
    def has_cache(self):
        return bool(self._cache)

    # =========================================================
    # UTILS
    # =========================================================

    def embed_text_with_values(self, text: str) -> dict[str, str | list[float]]:
        emb = self.embed_query(text)

        return {
            "text": text,
            "embedding": emb
        }

    def register_texts_with_embeddings(self, texts: list[str], embeddings: list[list[float]],):
        if len(texts) != len(embeddings):
            raise ValueError(
                "texts and embeddings must have identical length."
            )

        for text, emb in zip(texts, embeddings):
            self._cache[text] = emb

    def register_texts(self, texts: list[str]):

        uncached = [t for t in texts if t not in self._cache]

        if not uncached:
            return

        embeddings = self._encode_batch(uncached)

        for text, emb in zip(uncached, embeddings):
            self._cache[text] = emb
    
    def _require_model(self):
        if self.modele is None:
            self.load()

    def unload(self):
        if self.modele is None:
            return
        
        object.__setattr__(self, "modele", None,)

        self.clear_cache()
        clear_gpu()