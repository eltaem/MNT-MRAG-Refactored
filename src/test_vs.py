from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    host="localhost",
    port=6333,
)

print(client.get_collections())

client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    ),
)

print("Success")