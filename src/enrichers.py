
from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import BaseImageCaptioningModel, BaseTableCaptioningModel
from src.nodes import BaseNode, ImageNode, TableNode
from src.configs import BaseNodeEnricherConfig, ImageCaptionEnricherConfig, TableCaptionEnricherConfig
import asyncio
from src.utility import logger

class BaseNodeEnricher(ABC):
    """
    Transform:

        list[BaseNode]
            ↓
        list[BaseNode]

    Usually adds information:
        - image captions
        - OCR
        - metadata
        - table summaries
    """

    def __init__(
        self,
        config: BaseNodeEnricherConfig,
    ):
        self.config = config

    @abstractmethod
    def enrich(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:
        pass

    async def aenrich(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:

        return await asyncio.to_thread(
            self.enrich,
            nodes,
        )

class ImageCaptionEnricher(
    BaseNodeEnricher
):

    def __init__(
        self,
        config: ImageCaptionEnricherConfig,
        model: BaseImageCaptioningModel,
    ):
        super().__init__(config)

        self.config = config
        self.model = model

    def enrich(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:

        if not self.config.enabled:
            logger.info(
                "ImageCaptionEnricher disabled. Skipping."
            )
            return nodes

        caption_map: dict[str, str] = {}

        image_nodes = [
            node
            for node in nodes
            if isinstance(node, ImageNode)
            and not (
                self.config.skip_existing
                and node.caption is not None
            )
        ]

        logger.info(
            "Captioning %d image(s)...",
            len(image_nodes),
        )

        for i in range(
            0,
            len(image_nodes),
            self.config.batch_size,
        ):
            batch = image_nodes[
                i : i + self.config.batch_size
            ]

            logger.debug(
                "Processing image batch %d-%d",
                i + 1,
                min(
                    i + len(batch),
                    len(image_nodes),
                ),
            )

            for node in batch:

                logger.debug(
                    "Captioning image: %s",
                    node.image_path,
                )

                caption = self.model.caption(
                    node.image_path,
                )

                logger.debug(
                    "Caption: %s",
                    caption,
                )

                caption_map[node.node_id] = caption

        logger.info(
            "Finished captioning %d image(s).",
            len(caption_map),
        )

        enriched_nodes: list[BaseNode] = []

        for node in nodes:

            if (
                isinstance(node, ImageNode)
                and node.node_id in caption_map
            ):
                node = node.model_copy(
                    update={
                        "caption":
                            caption_map[
                                node.node_id
                            ]
                    }
                )

            enriched_nodes.append(node)

        return enriched_nodes

class TableCaptionEnricher(
    BaseNodeEnricher
):

    def __init__(
        self,
        config: TableCaptionEnricherConfig,
        model: BaseTableCaptioningModel,
    ):
        super().__init__(config)

        self.config = config
        self.model = model

    def enrich(
        self,
        nodes: list[BaseNode],
    ) -> list[BaseNode]:

        if not self.config.enabled:
            logger.info(
                "TableCaptionEnricher disabled. Skipping."
            )
            return nodes

        caption_map: dict[str, str] = {}

        table_nodes = [
            node
            for node in nodes
            if isinstance(node, TableNode)
            and not (
                self.config.skip_existing
                and node.caption is not None
            )
        ]

        logger.info(
            "Captioning %d table(s)...",
            len(table_nodes),
        )

        for i in range(
            0,
            len(table_nodes),
            self.config.batch_size,
        ):
            batch = table_nodes[
                i : i + self.config.batch_size
            ]
            
            logger.debug(
                "Processing table batch %d-%d",
                i + 1,
                min(
                    i + len(batch),
                    len(table_nodes),
                ),
            )

            for node in batch:

                logger.debug(
                    "Captioning table on page %d (%d rows × %d columns)",
                    node.page_number,
                    node.row_count,
                    node.column_count,
                )

                caption = self.model.caption(
                    node.rows
                )

                logger.debug(
                    "Generated table caption: %s",
                    caption,
                )

                caption_map[node.node_id] = caption

        logger.info(
            "Finished captioning %d table(s).",
            len(caption_map),
        )
        
        enriched_nodes: list[BaseNode] = []

        for node in nodes:

            if (
                isinstance(node, TableNode)
                and node.node_id in caption_map
            ):
                node = node.model_copy(
                    update={
                        "caption":
                            caption_map[
                                node.node_id
                            ]
                    }
                )

            enriched_nodes.append(node)

        return enriched_nodes