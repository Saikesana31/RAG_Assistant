# Loading and chunking the data
# using llama-index to load the pdf files and embedding the chunks

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_DIM =3072

# Load the pdf files and chunk the data
def load_and_chunk_data(pdf_path: str):
    reader = PDFReader()
    documents = reader.load_data(pdf_path)
    text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    texts = [doc.text for doc in documents if getattr(doc, "text", None)]
    chunks = []
    for i in texts:
        chunks.extend(text_splitter.split_text(i))
    return chunks


# enbed the chunks
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=chunks
    )
    return [item.embedding for item in response.data]
    