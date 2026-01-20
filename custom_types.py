import pydantic

# Represents chunked text data from a single document.
class RAGChunkAndSrc(pydantic.BaseModel):
    chunk: list[str]
    source_id: str = None

# Reports how many chunks were successfully stored in the vector database
class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

# Contains the retrieved chunks and their sources after a vector search.
class RAGSearchResult(pydantic.BaseModel):
    context: list[str]
    source: list[str]

# The final response to a user's question, including the answer and metadata.
class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources : list[str]
    num_contexts: int