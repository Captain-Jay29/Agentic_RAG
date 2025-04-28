# CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

import asyncio
import os
import logging
import json
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, basic_auth
import chromadb
from chromadb.utils import embedding_functions
import pprint # For pretty printing results

# --- Configuration ---
load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
COLLECTION_NAME = "backend_topics_gfg"
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection Setup ---
_neo4j_driver = None
_chromadb_client = None
_chromadb_collection = None
_chromadb_embedding_function = None

async def get_neo4j_driver():
    """Initializes and returns the async Neo4j driver instance."""
    global _neo4j_driver
    
    if _neo4j_driver is None:
        logging.info(f"Initializing Neo4j async driver for URI: {NEO4J_URL}")
        if not all([NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD]):
            logging.error("Neo4j connection details missing in environment variables.")
            raise ValueError("Missing Neo4j connection details")
        try:
            _neo4j_driver = AsyncGraphDatabase.driver(
                NEO4J_URL,
                auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
            )
            await _neo4j_driver.verify_connectivity()
            logging.info("Neo4j async driver initialized and connected.")
        except Exception as e:
            logging.error(f"Failed to initialize Neo4j async driver: {e}")
            _neo4j_driver = None # Reset on failure
            raise
    return _neo4j_driver

async def close_neo4j_driver():
    """Closes the global Neo4j driver if it exists."""
    global _neo4j_driver
    
    if _neo4j_driver:
        logging.info("Closing Neo4j async driver.")
        await _neo4j_driver.close()
        _neo4j_driver = None

def get_chromadb_collection():
    """Initializes ChromaDB client and returns the specific collection."""
    global _chromadb_client, _chromadb_collection, _chromadb_embedding_function
    
    if _chromadb_collection is None:
        logging.info(f"Initializing ChromaDB client (persistent path: ./{CHROMA_PERSIST_DIR})")
        if not os.path.exists(CHROMA_PERSIST_DIR):
             logging.error(f"ChromaDB persistence directory not found: {CHROMA_PERSIST_DIR}")
             raise FileNotFoundError(f"ChromaDB persistence directory not found: {CHROMA_PERSIST_DIR}")

        try:
            _chromadb_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            logging.info(f"Initializing Sentence Transformer embedding function ({MODEL_NAME}) for ChromaDB queries...")
            _chromadb_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
            logging.info(f"Getting ChromaDB collection: {COLLECTION_NAME}")
            _chromadb_collection = _chromadb_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=_chromadb_embedding_function
            )
            logging.info(f"ChromaDB collection '{COLLECTION_NAME}' retrieved.")

        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB client or get collection: {e}")
            _chromadb_client = None
            _chromadb_collection = None
            _chromadb_embedding_function = None
            raise
    return _chromadb_collection

# --- Neo4j Retrieval Functions ---
async def retrieve_topic_details_neo4j(topic_id: str) -> dict | None:
    """Fetches details for a specific topic from Neo4j."""
    logging.debug(f"Retrieving details for topic_id: {topic_id}")
    driver = await get_neo4j_driver()
    query = "MATCH (t:Topic {topic_id: $topic_id}) RETURN properties(t) AS details"
    try:
        result = await driver.execute_query(query, topic_id=topic_id, database_="neo4j")
        # Use .single() to get the first record or None safely
        record = result.records[0] if result.records else None # Adjusted for clarity
        summary = result.summary
        logging.debug(f"Query for topic {topic_id} took {summary.result_available_after}ms")

        if record:
            logging.debug(f"Found details for topic_id {topic_id}")
            return record['details'] # Returns the properties map
        else:
            logging.warning(f"Topic not found in Neo4j for topic_id: {topic_id}")
            return None
    except Exception as e:
        logging.error(f"Error querying Neo4j for topic details ({topic_id}): {e}")
        return None

async def get_next_topic_ids_neo4j(topic_id: str) -> list[str]:
    """Finds the topic_id(s) immediately following the given topic_id within the same section."""
    logging.debug(f"Getting next topic(s) for topic_id: {topic_id}")
    driver = await get_neo4j_driver()
    query = """
    MATCH (current:Topic {topic_id: $topic_id})-[:PRECEDES]->(next:Topic)
    RETURN next.topic_id AS next_topic_id
    """
    next_topic_ids = []
    try:
        result = await driver.execute_query(query, topic_id=topic_id, database_="neo4j")
        summary = result.summary
        logging.debug(f"Query for next topics ({topic_id}) took {summary.result_available_after}ms")
        for record in result.records:
            next_topic_ids.append(record["next_topic_id"])
        logging.debug(f"Next topic(s) for {topic_id}: {next_topic_ids}")
    except Exception as e:
        logging.error(f"Error querying Neo4j for next topics ({topic_id}): {e}")
    return next_topic_ids

async def get_previous_topic_id_neo4j(topic_id: str) -> str | None:
    """Finds the topic_id immediately preceding the given topic_id within the same section."""
    logging.debug(f"Getting previous topic for topic_id: {topic_id}")
    driver = await get_neo4j_driver()
    query = """
    MATCH (prev:Topic)-[:PRECEDES]->(current:Topic {topic_id: $topic_id})
    RETURN prev.topic_id AS prev_topic_id
    """
    try:
        result = await driver.execute_query(query, topic_id=topic_id, database_="neo4j")
        record = result.records[0] if result.records else None
        summary = result.summary
        logging.debug(f"Query for previous topic ({topic_id}) took {summary.result_available_after}ms")
        if record:
            prev_id = record["prev_topic_id"]
            logging.debug(f"Previous topic for {topic_id}: {prev_id}")
            return prev_id
        else:
             logging.debug(f"No previous topic found for {topic_id}")
             return None
    except Exception as e:
        logging.error(f"Error querying Neo4j for previous topic ({topic_id}): {e}")
        return None

async def get_topic_section_neo4j(topic_id: str) -> dict | None:
    """Finds the section containing the given topic_id."""
    logging.debug(f"Getting section for topic_id: {topic_id}")
    driver = await get_neo4j_driver()
    query = """
    MATCH (s:Section)-[:CONTAINS]->(t:Topic {topic_id: $topic_id})
    RETURN properties(s) AS section_details
    """
    try:
        result = await driver.execute_query(query, topic_id=topic_id, database_="neo4j")
        record = result.records[0] if result.records else None
        summary = result.summary
        logging.debug(f"Query for topic section ({topic_id}) took {summary.result_available_after}ms")
        if record:
            logging.debug(f"Section for topic {topic_id}: {record['section_details']}")
            return record['section_details']
        else:
            logging.warning(f"Could not find section containing topic: {topic_id}")
            return None
    except Exception as e:
        logging.error(f"Error querying Neo4j for topic's section ({topic_id}): {e}")
        return None

async def get_next_section_id_neo4j(section_id: str) -> str | None:
    """Finds the section_id immediately following the given section_id."""
    logging.debug(f"Getting next section for section_id: {section_id}")
    driver = await get_neo4j_driver()
    query = """
    MATCH (current:Section {section_id: $section_id})-[:PRECEDES]->(next:Section)
    RETURN next.section_id AS next_section_id
    """
    try:
        result = await driver.execute_query(query, section_id=section_id, database_="neo4j")
        record = result.records[0] if result.records else None
        summary = result.summary
        logging.debug(f"Query for next section ({section_id}) took {summary.result_available_after}ms")
        if record:
            next_id = record["next_section_id"]
            logging.debug(f"Next section for {section_id}: {next_id}")
            return next_id
        else:
             logging.debug(f"No next section found for {section_id}")
             return None
    except Exception as e:
        logging.error(f"Error querying Neo4j for next section ({section_id}): {e}")
        return None

# --- ChromaDB Retrieval Function ---

def _query_chroma_sync(collection: chromadb.Collection, query_text: str, n_results: int) -> dict | None:
    """Synchronous helper function to query ChromaDB."""
    # This function runs in a separate thread via asyncio.to_thread
    try:
        logging.debug(f"Executing ChromaDB query for: '{query_text}'")
        results = collection.query(
            query_texts=[query_text], # API expects a list of queries
            n_results=n_results,
            include=['metadatas', 'documents', 'distances'] # Include necessary fields
        )
        logging.debug("ChromaDB query executed successfully.")
        return results
    except Exception as e:
        # Log error from within the thread for clarity
        logging.error(f"ChromaDB query failed within thread for text '{query_text}': {e}")
        return None

async def find_similar_topics_chroma(query_text: str, n_results: int = 3) -> list[dict]:
    """
    Finds similar topics in ChromaDB based on semantic search using asyncio.to_thread.
    Returns a list of dictionaries, each containing topic info and distance.
    """
    logging.info(f"Finding {n_results} similar topics for query: '{query_text}'")
    similar_topics = []
    try:
        # Get collection (ensures client/collection are initialized)
        # This call itself is synchronous but quick if already initialized
        collection = get_chromadb_collection()

        # Run the synchronous query function in a separate thread
        results = await asyncio.to_thread(
            _query_chroma_sync, collection, query_text, n_results
        )

        # Process results if the query was successful and returned data
        if results and results.get('ids') and results['ids'][0]:
            # Results are structured per query text; we only sent one query.
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            documents = results['documents'][0] # The text that was embedded

            logging.debug(f"Processing {len(ids)} results from ChromaDB.")
            for i, topic_id in enumerate(ids):
                # Extract relevant info, especially topic_title from metadata
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                document = documents[i] if documents and i < len(documents) else ""
                distance = distances[i] if distances and i < len(distances) else float('inf')

                similar_topics.append({
                    "topic_id": topic_id,
                    "distance": distance, # Lower distance = more similar
                    "document": document, # Original text embedded
                    "metadata": metadata, # Full metadata
                    "topic_title": metadata.get("topic_title", "N/A") # Convenience access
                })
            logging.info(f"Formatted {len(similar_topics)} similar topics from ChromaDB.")

        else:
             logging.warning(f"No similar topics found or error in ChromaDB query for: '{query_text}'")

    except Exception as e:
        # Catch potential errors from thread execution or result processing
        logging.error(f"Error during ChromaDB similarity search: {e}", exc_info=True)

    # Sort results by distance (ascending) just in case ChromaDB doesn't guarantee it
    similar_topics.sort(key=lambda x: x['distance'])

    return similar_topics


# --- Hybrid Retrieval Logic ---
async def hybrid_retrieval(
    query_text: str,
    current_topic_id: str | None = None,
    n_chroma_results: int = 5, # Fetch slightly more semantic results initially
    n_final_results: int = 3   # Limit final combined output
    ) -> list[dict]:
    """
    Performs hybrid retrieval using semantic search (ChromaDB) and
    structural/detailed lookup (Neo4j). Returns a ranked list of enriched topic details.
    """
    logging.info(f"Performing hybrid retrieval for query: '{query_text}', current topic: {current_topic_id}")
    # Use dict keyed by topic_id for deduplication and enrichment
    # Value will store { 'details': neo4j_props, 'distance': chroma_distance, 'source': 'semantic'/'structural'/etc }
    combined_results = {}

    # --- Step 1: Concurrent Semantic Search and Current Topic Fetch (if applicable) ---
    tasks_to_gather = {}
    tasks_to_gather["semantic"] = find_similar_topics_chroma(query_text, n_results=n_chroma_results)

    if current_topic_id:
        tasks_to_gather["current_details"] = retrieve_topic_details_neo4j(current_topic_id)
        tasks_to_gather["current_prev_topic"] = get_previous_topic_id_neo4j(current_topic_id)
        tasks_to_gather["current_next_topics"] = get_next_topic_ids_neo4j(current_topic_id)
        tasks_to_gather["current_section"] = get_topic_section_neo4j(current_topic_id)

    try:
        # Run Chroma search and current topic lookups concurrently
        gathered_results = await asyncio.gather(*tasks_to_gather.values(), return_exceptions=True)
        results_map = dict(zip(tasks_to_gather.keys(), gathered_results))

        # Process results, handling potential errors from gather
        chroma_results = results_map.get("semantic")
        if isinstance(chroma_results, Exception):
            logging.error(f"ChromaDB search failed: {chroma_results}")
            chroma_results = [] # Treat as empty results if failed

        current_topic_details = results_map.get("current_details")
        if isinstance(current_topic_details, Exception):
            logging.error(f"Failed fetching current topic details: {current_topic_details}")
            current_topic_details = None

        # Store current topic details if found
        if current_topic_details:
            combined_results[current_topic_id] = {
                "details": current_topic_details,
                "distance": -1.0, # Indicate structural source primarily
                "source": "current_structural"
            }
            # Store neighbor info retrieved concurrently
            current_prev = results_map.get("current_prev_topic")
            current_next = results_map.get("current_next_topics")
            current_sect = results_map.get("current_section")
            if not isinstance(current_prev, Exception): combined_results[current_topic_id]["prev_topic_id"] = current_prev
            if not isinstance(current_next, Exception): combined_results[current_topic_id]["next_topic_ids"] = current_next
            if not isinstance(current_sect, Exception): combined_results[current_topic_id]["section_details"] = current_sect

            # Fetch next section if section details were retrieved
            if current_sect and not isinstance(current_sect, Exception) and current_sect.get("section_id"):
                next_sect_id = await get_next_section_id_neo4j(current_sect["section_id"])
                if next_sect_id: combined_results[current_topic_id]["next_section_id"] = next_sect_id

    except Exception as e:
        logging.error(f"Error during initial retrieval gather setup: {e}", exc_info=True)
        return [] # Return empty list if essential parts fail


    # --- Step 2: Enrich ChromaDB results with Neo4j details ---
    enrichment_tasks = {}
    valid_chroma_topic_ids = []
    chroma_results_map = {res['topic_id']: res for res in chroma_results if res.get('topic_id')} # Map for easy lookup

    for topic_id in chroma_results_map.keys():
        valid_chroma_topic_ids.append(topic_id)
        # Fetch details only if not already fetched as the current topic
        if topic_id not in combined_results:
            enrichment_tasks[topic_id] = retrieve_topic_details_neo4j(topic_id)

    # Fetch details for ChromaDB results concurrently
    if enrichment_tasks:
         try:
             enriched_details_results = await asyncio.gather(*enrichment_tasks.values(), return_exceptions=True)
             enriched_details_map = dict(zip(enrichment_tasks.keys(), enriched_details_results))

             # Add enriched data to combined_results
             for topic_id, details_result in enriched_details_map.items():
                 if isinstance(details_result, Exception) or not details_result:
                     logging.warning(f"Failed to enrich topic {topic_id} from Neo4j: {details_result}")
                     # Optionally add anyway using Chroma metadata as fallback
                     if topic_id not in combined_results:
                          chroma_res = chroma_results_map[topic_id]
                          combined_results[topic_id] = {
                              "details": chroma_res.get("metadata", {"title": chroma_res.get("topic_title", "N/A")}), # Basic fallback
                              "distance": chroma_res.get("distance", float('inf')),
                              "source": "semantic_unennriched"
                          }
                 elif topic_id not in combined_results: # Add successfully enriched result
                      chroma_res = chroma_results_map[topic_id]
                      combined_results[topic_id] = {
                          "details": details_result, # Use Neo4j details
                          "distance": chroma_res.get("distance", float('inf')),
                          "source": "semantic_enriched"
                      }
                 # Note: If topic_id was already present from 'current_topic', we don't overwrite details
                 # but we could update the 'source' and 'distance' if desired. Let's keep it simple for now.

         except Exception as e:
             logging.error(f"Error during enrichment gather: {e}", exc_info=True)
             # Proceed with potentially less enriched data


    # --- Step 3: Format and Rank Output ---
    # Convert dictionary back to list
    final_list = [{"topic_id": tid, **data} for tid, data in combined_results.items()]

    # Simple ranking: Lower distance is better. Give current topic a slight edge if found semantically.
    def sort_key(item):
        is_current = item.get("topic_id") == current_topic_id
        # Boost current topic slightly (lower distance effective value)
        # Give non-semantic results (structural only) a higher distance
        effective_distance = item.get("distance", float('inf'))
        if is_current and "semantic" in item.get("source",""): # Current topic also found semantically
             effective_distance = min(effective_distance, 0.1) # Give it a very good score
        elif is_current:
             effective_distance = 0.0 # Best score if only structural match
        elif effective_distance == -1.0: # If only structural, penalize slightly vs semantic
            effective_distance = float('inf') # Push non-semantic matches further down

        return effective_distance

    final_list.sort(key=sort_key)

    logging.info(f"Hybrid retrieval generated {len(final_list)} combined results. Limiting to {n_final_results}.")

    # Limit to final number of results
    return final_list[:n_final_results]



# --- Main function for testing ---
async def main_test():
    logging.info("--- Starting Retrieval Logic Tests ---")
    neo4j_driver = None
    try:
        # Test Connections
        logging.info("Testing database connections...")
        neo4j_driver = await get_neo4j_driver()
        collection = await asyncio.to_thread(get_chromadb_collection)
        logging.info(f"ChromaDB collection count: {collection.count()}")
        logging.info("Database connections successful.")

        # --- Test Neo4j Functions ---
        logging.info("\n\n--- Testing Neo4j Functions (Summary) ---")
        assert await retrieve_topic_details_neo4j("apis_rest") is not None
        assert await get_next_topic_ids_neo4j("apis_json") == ['apis_graphql']
        assert await get_previous_topic_id_neo4j("apis_graphql") == 'apis_json'
        logging.info("Neo4j function basic tests passed.")


        # --- Test ChromaDB Retrieval Function ---
        logging.info("\n\n--- Testing ChromaDB Function (Summary) ---")
        assert len(await find_similar_topics_chroma("strategies for caching data", n_results=1)) > 0
        logging.info("ChromaDB function basic test passed.")


        # --- Test Hybrid Retrieval Function ---
        logging.info("\n\n--- Testing Hybrid Retrieval Function ---")

        # Test Case 1: General query, no current topic
        query1 = "How to handle database connections?"
        logging.info(f"\n\nHybrid Test 1: Query='{query1}', Current=None")
        results1 = await hybrid_retrieval(query_text=query1)
        pprint.pprint(results1)

        # Test Case 2: Query related to a specific area, with current topic
        query2 = "Testing APIs"
        current_topic2 = "apis_rest" # Where the user currently is
        logging.info(f"\n\nHybrid Test 2: Query='{query2}', Current='{current_topic2}'")
        results2 = await hybrid_retrieval(query_text=query2, current_topic_id=current_topic2)
        pprint.pprint(results2)

        # Test Case 3: Query directly asking for next step from current topic
        query3 = "What comes after relational databases?"
        current_topic3 = "databases_relational"
        logging.info(f"\n\nHybrid Test 3: Query='{query3}', Current='{current_topic3}'")
        results3 = await hybrid_retrieval(query_text=query3, current_topic_id=current_topic3)
        pprint.pprint(results3)

        # Test Case 4: Query semantically similar to current topic
        query4 = "Ways to authenticate API requests" # Similar to 'apis_authentication'
        current_topic4 = "apis_authentication"
        logging.info(f"\n\nHybrid Test 4: Query='{query4}', Current='{current_topic4}'")
        results4 = await hybrid_retrieval(query_text=query4, current_topic_id=current_topic4)
        pprint.pprint(results4)


        logging.info("\n\n--- Hybrid Retrieval Tests Complete ---")


    except Exception as e:
        logging.error(f"An error occurred during testing: {e}", exc_info=True)
    finally:
        await close_neo4j_driver()
        logging.info("--- Retrieval Logic Tests Finished ---")


if __name__ == "__main__":
    asyncio.run(main_test())