import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app.ingest")


def _env(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in {'"', "'"}):
        val = val[1:-1]
    return val or None


def main() -> None:
    # Print immediately so we can see if we're stuck on imports.
    print("ingest: starting (before heavy imports)", flush=True)

    logger.info("Importing LangChain/Pinecone deps")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    # ---- Load docs ----
    logger.info("Loading documents")
    loader = DirectoryLoader("data", glob="**/*.txt")
    documents = loader.load()

    # ---- Chunking ----
    logger.info("Chunking documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.split_documents(documents)

    # ---- Embeddings ----
    logger.info("Initializing embeddings (may download model on first run)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # ---- Pinecone init ----
    logger.info("Initializing Pinecone client")
    pc = Pinecone(api_key=_env("PINECONE_API_KEY"))

    index_name = _env("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is required")

    # Create index if not exists
    logger.info("Checking/creating Pinecone index: %s", index_name)
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,  # MiniLM embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=_env("PINECONE_ENV") or "us-east-1",
            ),
        )

    index = pc.Index(index_name)

    # ---- Store vectors ----
    logger.info("Uploading %d chunks to Pinecone", len(docs))
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    vectorstore.add_documents(docs)
    logger.info("Done: data indexed in Pinecone")


if __name__ == "__main__":
    main()
