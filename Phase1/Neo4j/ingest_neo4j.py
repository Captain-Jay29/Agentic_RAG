import json
import os
import logging
from neo4j import GraphDatabase, basic_auth # type: ignore
from dotenv import load_dotenv # type: ignore

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL") 
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# File containing the structured roadmap data
ROADMAP_JSON_PATH = "/Users/jay/Desktop/The File/Learn/Agentic_RAG_mem0/Phase1/backend_developer_roadmap_gfg.json"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Neo4j Transaction Functions ---

def _create_constraints(tx):
    """Creates necessary constraints if they don't exist."""
    commands = [
        "CREATE CONSTRAINT unique_roadmap_title IF NOT EXISTS FOR (r:Roadmap) REQUIRE r.title IS UNIQUE;",
        "CREATE CONSTRAINT unique_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE;",
        "CREATE CONSTRAINT unique_topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE;",
        # Add user/resource constraints later if needed
    ]
    for command in commands:
        try:
            logging.info(f"Running: {command}")
            tx.run(command)
        except Exception as e:
            # Ignore errors if constraint already exists, log others
            if "already exists" not in str(e):
                logging.error(f"Failed to run constraint command '{command}': {e}")
            else:
                logging.warning(f"Constraint likely already exists for command: {command}")


def _create_indexes(tx):
    """Creates recommended indexes if they don't exist."""
    commands = [
        "CREATE INDEX section_title_index IF NOT EXISTS FOR (s:Section) ON (s.title);",
        "CREATE INDEX topic_title_index IF NOT EXISTS FOR (t:Topic) ON (t.title);",
        "CREATE INDEX topic_is_choice_index IF NOT EXISTS FOR (t:Topic) ON (t.is_choice);",
        "CREATE INDEX section_order_index IF NOT EXISTS FOR (s:Section) ON (s.order);",
        "CREATE INDEX topic_order_index IF NOT EXISTS FOR (t:Topic) ON (t.order);",
    ]
    for command in commands:
        try:
            logging.info(f"Running: {command}")
            tx.run(command)
        except Exception as e:
             # Ignore errors if index already exists, log others
            if "already exists" not in str(e):
                logging.error(f"Failed to run index command '{command}': {e}")
            else:
                 logging.warning(f"Index likely already exists for command: {command}")

def _create_roadmap_node(tx, title, source_url):
    """Creates the main Roadmap node."""
    query = """
    MERGE (r:Roadmap {title: $title})
    ON CREATE SET r.source_url = $source_url
    ON MATCH SET r.source_url = $source_url // Update source_url if node exists
    RETURN r.title
    """
    result = tx.run(query, title=title, source_url=source_url)
    logging.info(f"Ensured Roadmap node exists with title: {title}")
    return result.single()[0]

def _create_section_node(tx, section_data, roadmap_title):
    """Creates a Section node and links it to the Roadmap."""
    # Create/Merge Section Node
    query_section = """
    MERGE (s:Section {section_id: $section_id})
    ON CREATE SET s.title = $title, s.order = $order
    ON MATCH SET s.title = $title, s.order = $order // Update properties if node exists
    RETURN s.section_id
    """
    tx.run(query_section,
           section_id=section_data['section_id'],
           title=section_data['section_title'],
           order=section_data['order'])

    # Link Section to Roadmap
    query_link = """
    MATCH (r:Roadmap {title: $roadmap_title})
    MATCH (s:Section {section_id: $section_id})
    MERGE (r)-[:CONTAINS]->(s)
    """
    tx.run(query_link, roadmap_title=roadmap_title, section_id=section_data['section_id'])
    logging.info(f"  Ensured Section node exists: {section_data['section_id']} ({section_data['section_title']}) and linked to Roadmap.")

def _create_topic_node(tx, topic_data, section_id):
    """Creates a Topic node and links it to its Section."""
    options = topic_data.get('options', [])
    is_choice = bool(options) # True if options list is not empty

    # Create/Merge Topic Node
    query_topic = """
    MERGE (t:Topic {topic_id: $topic_id})
    ON CREATE SET t.title = $title,
                  t.order = $order,
                  t.description = $description,
                  t.options = $options,
                  t.is_choice = $is_choice
    ON MATCH SET  t.title = $title,
                  t.order = $order,
                  t.description = $description,
                  t.options = $options,
                  t.is_choice = $is_choice // Update properties if node exists
    RETURN t.topic_id
    """
    tx.run(query_topic,
           topic_id=topic_data['topic_id'],
           title=topic_data['topic_title'],
           order=topic_data['order'],
           description=topic_data.get('description', ''),
           options=options,
           is_choice=is_choice)

    # Link Topic to Section
    query_link = """
    MATCH (s:Section {section_id: $section_id})
    MATCH (t:Topic {topic_id: $topic_id})
    MERGE (s)-[:CONTAINS]->(t)
    """
    tx.run(query_link, section_id=section_id, topic_id=topic_data['topic_id'])
    logging.info(f"    Ensured Topic node exists: {topic_data['topic_id']} ({topic_data['topic_title']}) and linked to Section {section_id}.")

def _create_precedes_relationships(tx, sections_data):
    """Creates PRECEDES relationships between sections and topics."""
    logging.info("Creating PRECEDES relationships...")
    # --- Link Sections ---
    # Sort sections by order
    sorted_sections = sorted(sections_data, key=lambda s: s['order'])
    section_query = """
    MATCH (s1:Section {section_id: $section_id_1})
    MATCH (s2:Section {section_id: $section_id_2})
    MERGE (s1)-[r:PRECEDES]->(s2)
    SET r.order_diff = 1 // Optional: add property if needed
    """
    for i in range(len(sorted_sections) - 1):
        s1 = sorted_sections[i]
        s2 = sorted_sections[i+1]
        tx.run(section_query, section_id_1=s1['section_id'], section_id_2=s2['section_id'])
        logging.info(f"  Linked Section {s1['section_id']} -[:PRECEDES]-> Section {s2['section_id']}")

    # --- Link Topics within each Section ---
    topic_query = """
    // Match topics within the same section based on IDs
    MATCH (s:Section {section_id: $section_id})
    MATCH (t1:Topic {topic_id: $topic_id_1})
    MATCH (t2:Topic {topic_id: $topic_id_2})
    // Ensure they belong to the same section (redundant check, but safe)
    WHERE (s)-[:CONTAINS]->(t1) AND (s)-[:CONTAINS]->(t2)
    // Create relationship
    MERGE (t1)-[r:PRECEDES]->(t2)
    SET r.order_diff = 1 // Optional: add property if needed
    """
    for section in sections_data:
        section_id = section['section_id']
        # Sort topics by order within the section
        sorted_topics = sorted(section.get('topics', []), key=lambda t: t['order'])
        for i in range(len(sorted_topics) - 1):
            t1 = sorted_topics[i]
            t2 = sorted_topics[i+1]
            tx.run(topic_query, section_id=section_id, topic_id_1=t1['topic_id'], topic_id_2=t2['topic_id'])
            logging.info(f"    Linked Topic {t1['topic_id']} -[:PRECEDES]-> Topic {t2['topic_id']} in Section {section_id}")

    logging.info("Finished creating PRECEDES relationships.")

# --- Main Execution ---
def main():
    # --- 1. Check Environment Variables ---
    if not all([NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD]):
        logging.error("Neo4j connection details (NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD) not found in environment variables.")
        logging.error("Please ensure they are set in your .env file or environment.")
        return

    # --- 2. Load Roadmap Data ---
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

    # --- 3. Connect to Neo4j ---
    logging.info(f"Connecting to Neo4j AuraDB at {NEO4J_URL}...")
    driver = None # Initialize driver variable
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logging.info("Successfully connected to Neo4j.")
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        return # Exit if connection fails

    # --- 4. Ingest Data within Sessions/Transactions ---
    try:
        # Get database name (AuraDB Free uses 'neo4j' by default)
        # You might need to specify if using a different DB: database="your_db_name"
        with driver.session(database="neo4j") as session:
            # --- Create Constraints & Indexes (run once ideally) ---
            # It's safe to run these each time due to "IF NOT EXISTS"
            logging.info("Ensuring constraints and indexes exist...")
            session.execute_write(_create_constraints)
            session.execute_write(_create_indexes)
            logging.info("Constraints and indexes checked/created.")

            # --- Create Roadmap Node ---
            roadmap_title = roadmap_data.get("roadmap_title", "Unknown Roadmap")
            source_url = roadmap_data.get("source_url", "")
            session.execute_write(_create_roadmap_node, roadmap_title, source_url)

            # --- Create Section and Topic Nodes + CONTAINS Links ---
            sections_data = roadmap_data.get("sections", [])
            if not sections_data:
                 logging.warning("No sections found in the JSON data.")
            else:
                logging.info(f"Processing {len(sections_data)} sections...")
                for section in sections_data:
                    session.execute_write(_create_section_node, section, roadmap_title)
                    topics_data = section.get("topics", [])
                    if not topics_data:
                         logging.warning(f"No topics found in section: {section['section_id']}")
                    else:
                        logging.info(f"  Processing {len(topics_data)} topics for section {section['section_id']}...")
                        for topic in topics_data:
                            session.execute_write(_create_topic_node, topic, section['section_id'])

                # --- Create PRECEDES Relationships ---
                # Pass the list of section data to the function
                session.execute_write(_create_precedes_relationships, sections_data)

        logging.info("Neo4j ingestion completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during Neo4j ingestion: {e}")
    finally:
        # --- 5. Close Driver Connection ---
        if driver:
            logging.info("Closing Neo4j driver connection.")
            driver.close()

if __name__ == "__main__":
    main()