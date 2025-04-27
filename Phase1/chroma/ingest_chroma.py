import json
import chromadb # type: ignore
from chromadb.utils import embedding_functions # type: ignore # Import Chroma's helper
import os
from tqdm import tqdm # type: ignore # For progress bars
import logging

# --- Configuration ---
ROADMAP_JSON_PATH = "/Users/jay/Desktop/The File/Learn/Agentic_RAG_mem0/Phase1/backend_developer_roadmap_gfg.json"
CHROMA_PERSIST_DIR = "chroma_db_backend_gfg"
COLLECTION_NAME = "backend_topics_gfg"

MODEL_NAME = 'all-MiniLM-L6-v2' # Sentence Transformer model name
BATCH_SIZE = 32

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- 1. Load Roadmap Data ---
    logging.info(f"Loading roadmap data from {ROADMAP_JSON_PATH}...")
    try:
        with open(ROADMAP_JSON_PATH, 'r', encoding='utf-8') as f:
            roadmap_data = json.load(f)
        logging.info("Roadmap data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Roadmap file not found at {ROADMAP_JSON_PATH}")
        return
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {ROADMAP_JSON_PATH}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred loading the file: {e}")
        return

    # --- 2. Initialize ChromaDB Client ---
    logging.info(f"Initializing ChromaDB client (persisting to ./{CHROMA_PERSIST_DIR})...")
    if not os.path.exists(CHROMA_PERSIST_DIR):
         os.makedirs(CHROMA_PERSIST_DIR) # Create directory if it doesn't exist
         logging.info(f"Created persistence directory: {CHROMA_PERSIST_DIR}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logging.info("ChromaDB client initialized.")
    except Exception as e:
        logging.error(f"Error initializing ChromaDB client: {e}")
        return

    # --- 3. Initialize Embedding Function ---
    # Use ChromaDB's built-in helper for Sentence Transformers
    logging.info(f"Initializing Sentence Transformer embedding function ({MODEL_NAME})...")
    try:
        # This helper function handles loading the model
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
            # Optionally specify device: device="cuda" or device="cpu"
            # device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info("Embedding function initialized.")
    except Exception as e:
        logging.error(f"Error initializing Sentence Transformer model: {e}")
        logging.error("Make sure 'sentence-transformers' and a backend ('torch', 'tensorflow', 'jax') are installed.")
        return

    # --- 4. Get or Create ChromaDB Collection ---
    logging.info(f"Getting or creating ChromaDB collection: {COLLECTION_NAME}...")
    try:
        # Pass the embedding function instance directly
        # Use cosine distance as it's generally preferred for sentence embeddings
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info(f"Collection '{COLLECTION_NAME}' ready.")
    except Exception as e:
        logging.error(f"Error getting or creating collection: {e}")
        return

    # --- 5. Prepare Data for ChromaDB ---
    logging.info("Preparing documents, metadata, and IDs for ChromaDB...")
    documents_to_process = [] # Store as tuples (doc_content, metadata, topic_id)

    skipped_topics = 0
    processed_topic_ids = set()

    for section in roadmap_data.get("sections", []):
        section_id = section.get("section_id", "unknown_section")
        section_title = section.get("section_title", "Unknown Section")
        section_order = section.get("order", -1)

        for topic in section.get("topics", []):
            topic_id = topic.get("topic_id")
            topic_title = topic.get("topic_title", "Unknown Topic")
            description = topic.get("description", "")
            order_in_section = topic.get("order", -1)
            options = topic.get("options", []) # Get options if they exist

            # Basic validation
            if not topic_id:
                logging.warning(f"Skipping topic with missing 'topic_id' in section '{section_title}'")
                skipped_topics += 1
                continue

            if topic_id in processed_topic_ids:
                 logging.warning(f"Duplicate topic_id '{topic_id}' found. Skipping subsequent occurrence.")
                 skipped_topics +=1
                 continue

            # Combine title and description for embedding
            doc_content = f"{topic_title}: {description}".strip()
            if not doc_content or doc_content == ":":
                 logging.warning(f"Topic '{topic_id}' has empty title/description. Skipping embedding.")
                 skipped_topics += 1
                 continue

            metadata = {
                "topic_id": topic_id,
                "section_id": section_id,
                "topic_title": topic_title,
                "section_title": section_title,
                "order_in_section": order_in_section,
                "section_order": section_order,
                # Store options as a JSON string if needed, or handle separately
                "options_json": json.dumps(options) if options else ""
            }
            documents_to_process.append((doc_content, metadata, topic_id))
            processed_topic_ids.add(topic_id)


    logging.info(f"Prepared {len(documents_to_process)} unique documents for potential ingestion.")
    if skipped_topics > 0:
        logging.warning(f"Skipped {skipped_topics} topics due to missing ID, duplicate ID, or empty content.")

    if not documents_to_process:
        logging.info("No valid documents found to add to the collection. Exiting.")
        return

    # --- 6. Check Existing Documents & Add New Ones ---
    all_ids = [item[2] for item in documents_to_process]

    logging.info(f"Checking for {len(all_ids)} documents in the collection '{COLLECTION_NAME}'...")
    try:
        existing_data = collection.get(ids=all_ids)
        existing_ids = set(existing_data['ids'])
        logging.info(f"Found {len(existing_ids)} documents already existing in the collection.")
    except Exception as e:
         logging.error(f"Error checking for existing documents: {e}. Proceeding assuming none exist, duplicates may occur.")
         existing_ids = set() # Assume none exist if check fails

    # Filter out documents that already exist
    new_documents = []
    new_metadatas = []
    new_ids = []
    for doc_content, metadata, topic_id in documents_to_process:
         if topic_id not in existing_ids:
             new_documents.append(doc_content)
             new_metadatas.append(metadata)
             new_ids.append(topic_id)

    # Add new documents if any
    if not new_ids:
        logging.info("No new documents to add. All prepared documents already exist in the collection.")
    else:
        logging.info(f"Adding {len(new_ids)} new documents to the collection '{COLLECTION_NAME}'...")
        try:
            num_batches = (len(new_ids) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in tqdm(range(num_batches), desc="Adding batches to ChromaDB"):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, len(new_ids))

                batch_ids = new_ids[start_idx:end_idx]
                batch_docs = new_documents[start_idx:end_idx]
                batch_metas = new_metadatas[start_idx:end_idx]

                # Embeddings are generated implicitly by ChromaDB via the embedding_function
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
            logging.info(f"Successfully added {len(new_ids)} new documents to the collection.")

        except Exception as e:
            logging.error(f"Error adding documents to ChromaDB collection: {e}")
            # Consider adding more specific error handling (e.g., network issues, data format errors)

    # --- 7. Final Verification ---
    try:
        final_count = collection.count()
        logging.info(f"Collection '{COLLECTION_NAME}' now contains {final_count} documents.")
    except Exception as e:
        logging.warning(f"Could not verify final count in collection: {e}")

    logging.info("ChromaDB ingestion script finished.")

if __name__ == "__main__":
    main()