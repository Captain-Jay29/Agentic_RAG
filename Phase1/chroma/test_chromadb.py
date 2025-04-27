import chromadb # type: ignore
from chromadb.utils import embedding_functions # type: ignore
import json

CHROMA_PERSIST_DIR = "chroma_db_backend_gfg"
COLLECTION_NAME = "backend_topics_gfg"
MODEL_NAME = 'all-MiniLM-L6-v2' # Must match the model used for ingestion

try:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Get embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    # Get the collection (use the embedding function used during creation)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef # Important for query embedding
    )

    print(f"Successfully connected to collection '{COLLECTION_NAME}'.")
    print(f"Number of items in collection: {collection.count()}")

    # Optional: Peek at a few items
    print("\nPeeking at a few items:")
    print(collection.peek(limit=3))

    # Optional: Perform a sample query
    print("\nPerforming sample query for 'web security':")
    results = collection.query(
        query_texts=["web security concepts"],
        n_results=3,
        include=['metadatas', 'documents', 'distances'] # Include distances
    )
    print(json.dumps(results, indent=2))

except Exception as e:
    print(f"An error occurred during verification: {e}")