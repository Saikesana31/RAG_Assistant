# Vector Database - upload the vectors and search the vectors

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct


class QdrantStorage:
    # Initialize the Qdrant client and create a collection if it doesn't exist
    def __init__(self, url="http://localhost:6333", collection_name: str = "rag_collection",dim = 3072):
        self.client = QdrantClient(url = url, timeout=30)
        self.collection_name = collection_name
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    # Add vectors to the collection
    def upsert(self, ids , vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # Search the collection
    def search(self, query: str, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            with_payload=True,
            query=query,
            limit=top_k
        )

        context = []
        source = set()

        for result in results:
            payload = getattr(result, "payload", {})
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text :
                context.append(text)
                source.add(source)

        return {"context": context, "source": list(source)}