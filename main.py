import logging
from turtle import done
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import datetime
import uuid
from data_loader import load_and_chunk_data, embed_chunks
from vector_DB import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult


load_dotenv()

# Create an Inngest client - is used to send events to an inngest server
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# Create an Inngest function - is used to process events from an inngest server
@inngest_client.create_function(
    fn_id=" RAG: Ingest pdf files ",

    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context) -> str:
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id",pdf_path)
        chunks = load_and_chunk_data(pdf_path)
        return RAGChunkAndSrc(chunk=chunks, source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunk
        source_id = chunk_and_src.source_id
        vectors = embed_chunks(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id,"text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load_chunks", lambda : _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embeddings_upsert", lambda : _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger = inngest.TriggerEvent(event="rag/query_pdf")
)


async def rag_query_pdf(ctx: inngest.Context):
    def _search(question: str,top_k: int = 5)-> RAGSearchResult:
        query = embed_chunks([question])[0]
        results = QdrantStorage().search(query, top_k)
        return RAGSearchResult(context=results["context"], source=results["source"])

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    search_result = await ctx.step.run("embbed_search", lambda : _search(question, top_k), output_type=RAGSearchResult)
    
    context_block = "\n".join(f" - {c}" for c in search_result.context)
    user_content = (
        f"""
        use the context to answer the question in simple ways that is easy to understand by a Entry level AI engineer
        Context: {context_block}
        Question: {question}
        Answer: Concisely using the context above.
        """
    )

    
    adapter  = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )


    res = await ctx.step.ai.infer(
        "call-openai",
        adapter=adapter,
        # body is the model request
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                
                {"role": "system","content": "you are a helpful assistant that can answer questions based on the context provided"},
                {"role":"user","content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": search_result.source, "num_contexts": len(search_result.context)}


app = FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf])