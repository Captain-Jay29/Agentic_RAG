import sys
import os
import logging # Make sure logging is imported early too

# --- Dynamically Add Phase2 Directory to Path ---
try:
    # Calculate the absolute path to the Phase2 directory
    # Assumes agent_tools.py is in Phase3, and Phase2 is a sibling directory
    current_dir = os.path.dirname(__file__)
    phase2_dir = os.path.abspath(os.path.join(current_dir, '..', 'Phase2'))

    # Add Phase2 directory to sys.path if it's not already there
    if phase2_dir not in sys.path:
        sys.path.insert(0, phase2_dir)
        logging.info(f"Added {phase2_dir} to sys.path for imports.")

    # Now try importing from retrieval_logic
    from retrieval_logic import (
        hybrid_retrieval,
        retrieve_topic_details_neo4j,
        get_next_topic_ids_neo4j,
        get_previous_topic_id_neo4j,
        get_topic_section_neo4j,
        get_next_section_id_neo4j,
        # We still assume retrieval_logic manages its own connections
        # or connection objects need to be passed differently.
    )
    retrieval_logic_available = True
    logging.info("Successfully imported from retrieval_logic.")

except ImportError as e:
    logging.error(f"CRITICAL: Could not import from retrieval_logic.py even after modifying path: {e}")
    logging.error(f"Attempted to add path: {phase2_dir}")
    logging.error(f"Current sys.path: {sys.path}")
    retrieval_logic_available = False
    # Define dummy async functions if import fails
    async def hybrid_retrieval(*args, **kwargs): return [{"error": "retrieval_logic not found"}]
    async def retrieve_topic_details_neo4j(*args, **kwargs): return None
    async def get_next_topic_ids_neo4j(*args, **kwargs): return []
    async def get_previous_topic_id_neo4j(*args, **kwargs): return None
    async def get_topic_section_neo4j(*args, **kwargs): return None
    async def get_next_section_id_neo4j(*args, **kwargs): return None
except Exception as e:
    logging.error(f"An unexpected error occurred during path modification or import: {e}")
    retrieval_logic_available = False
    # Define dummy functions as above


# --- Continue with the rest of agent_tools.py ---
# (Keep all the Pydantic schemas and Tool classes as they were)
import asyncio
import json
from typing import Type, Optional, List # Already likely imported by BaseTool/Pydantic

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class HybridSearchInput(BaseModel):
    """Input schema for the roadmap_hybrid_search tool."""
    query_text: str = Field(description="The user's natural language query about the roadmap.")
    current_topic_id: Optional[str] = Field(None, description="Optional: The unique ID of the topic the user is currently focused on, if relevant to the query.")

class TopicDetailsInput(BaseModel):
    """Input schema for the get_topic_details tool."""
    topic_id: str = Field(description="The unique ID of the topic to get details for.")

class CurrentTopicInput(BaseModel):
    """Input schema for tools needing the user's current topic."""
    current_topic_id: str = Field(description="The unique ID of the topic the user is currently focused on or asking about sequentially.")

# --- Custom LangChain Tools ---

class HybridRoadmapSearchTool(BaseTool):
    """Tool to search the roadmap using hybrid retrieval."""
    name: str = "roadmap_hybrid_search"
    description: str = (
        "Use this tool to search the backend development roadmap when the user asks general questions "
        "about topics, concepts, tools, or asks for related information. It uses both semantic search "
        "and graph structure. Input should be the user's query text. You can also provide the 'current_topic_id' "
        "if the user's query seems related to their current position in the roadmap."
    )
    args_schema: Type[BaseModel] = HybridSearchInput

    async def _arun(self, query_text: str, current_topic_id: Optional[str] = None) -> str:
        """Use the tool asynchronously."""
        if not retrieval_logic_available: return "Error: Retrieval logic unavailable."
        logging.info(f"Tool '{self.name}' executing with query: '{query_text}', current_topic: {current_topic_id}")
        try:
            results = await hybrid_retrieval(
                query_text=query_text,
                current_topic_id=current_topic_id,
                n_final_results=3 # Agent might need top 3 relevant items
            )
            if not results:
                return "No relevant topics found in the roadmap for that query."

            # Format results into a string for the LLM agent
            formatted_results = []
            for i, res in enumerate(results):
                details = res.get("details", {})
                topic_id = res.get("topic_id", details.get("topic_id", "N/A"))
                title = details.get("title", "N/A")
                desc = details.get("description", "No description available.")
                options = details.get("options", [])
                distance = res.get("distance", -1.0) # Chroma distance if available

                result_str = f"{i+1}. Topic: '{title}' (ID: {topic_id})\n   Description: {desc}"
                if options:
                    result_str += f"\n   Options/Tools: {', '.join(options)}"
                # Optionally add distance if useful for agent reasoning
                # if distance != -1.0: result_str += f"\n (Semantic Distance: {distance:.4f})"
                formatted_results.append(result_str)

            return "Found the following relevant topics based on the query:\n" + "\n---\n".join(formatted_results)

        except Exception as e:
            logging.error(f"Error in {self.name} tool: {e}", exc_info=True)
            return "Sorry, there was an error searching the roadmap."

    def _run(self, *args, **kwargs): # Synchronous version (not recommended here)
         raise NotImplementedError(f"{self.name} does not support synchronous execution. Use `arun` instead.")


class GetTopicDetailsTool(BaseTool):
    """Tool to get details for a specific topic ID."""
    name: str = "get_topic_details"
    description: str = (
        "Use this tool when you need detailed information about a *specific* topic and you "
        "already know its unique topic ID. Input must be the 'topic_id'."
    )
    args_schema: Type[BaseModel] = TopicDetailsInput

    async def _arun(self, topic_id: str) -> str:
        """Use the tool asynchronously."""
        if not retrieval_logic_available: return "Error: Retrieval logic unavailable."
        logging.info(f"Tool '{self.name}' executing for topic_id: {topic_id}")
        try:
            details = await retrieve_topic_details_neo4j(topic_id)
            if not details:
                return f"No details found for topic ID '{topic_id}'. It might be an invalid ID."

            # Format details into a string for the LLM agent
            title = details.get('title', 'N/A')
            desc = details.get('description', 'No description.')
            options = details.get('options', [])
            is_choice = details.get('is_choice', False)

            result_str = f"Details for Topic '{title}' (ID: {topic_id}):\nDescription: {desc}"
            if is_choice and options:
                result_str += f"\nThis topic presents choices. Options/Tools: {', '.join(options)}"

            return result_str

        except Exception as e:
            logging.error(f"Error in {self.name} tool: {e}", exc_info=True)
            return f"Sorry, an error occurred while retrieving details for topic {topic_id}."

    def _run(self, *args, **kwargs): raise NotImplementedError(f"{self.name} does not support sync execution.")


class GetNextStepsTool(BaseTool):
    """Tool to find the next sequential step(s) in the roadmap."""
    name: str = "get_next_steps"
    description: str = (
        "Use this tool when the user asks 'What should I learn next?', 'What comes after X?', "
        "or wants to know the next step in the learning path from their current position. "
        "Input must be the 'current_topic_id'."
    )
    args_schema: Type[BaseModel] = CurrentTopicInput

    async def _arun(self, current_topic_id: str) -> str:
        """Use the tool asynchronously."""
        if not retrieval_logic_available: return "Error: Retrieval logic unavailable."
        logging.info(f"Tool '{self.name}' executing for current_topic_id: {current_topic_id}")
        try:
            # Fetch next topics and current section info concurrently
            next_topic_ids_task = get_next_topic_ids_neo4j(current_topic_id)
            current_section_task = get_topic_section_neo4j(current_topic_id)
            next_topic_ids, current_section = await asyncio.gather(
                next_topic_ids_task, current_section_task, return_exceptions=True
            )

            # Handle potential errors from gather
            if isinstance(next_topic_ids, Exception):
                logging.error(f"Failed fetching next topics for {current_topic_id}: {next_topic_ids}")
                next_topic_ids = []
            if isinstance(current_section, Exception):
                 logging.error(f"Failed fetching section for {current_topic_id}: {current_section}")
                 current_section = None

            response_parts = []
            if next_topic_ids:
                # Fetch details for next topics to show titles (best effort)
                next_topic_details_tasks = [retrieve_topic_details_neo4j(tid) for tid in next_topic_ids]
                next_topic_details_results = await asyncio.gather(*next_topic_details_tasks, return_exceptions=True)
                next_topic_titles = [
                    # Use title from details if fetch worked, else just use ID
                    res.get('title', tid) if res and not isinstance(res, Exception) else tid
                    for tid, res in zip(next_topic_ids, next_topic_details_results)
                ]
                topics_str = ', '.join(f"'{title}' (ID: {tid})" for tid, title in zip(next_topic_ids, next_topic_titles))
                response_parts.append(f"The next recommended topic(s) in this section are: {topics_str}.")
            else:
                # No more topics in this section, find the next section
                response_parts.append("You've reached the end of the topics in this section.")
                if current_section:
                    section_id = current_section.get("section_id")
                    section_title = current_section.get("title", section_id)
                    response_parts[-1] = f"You've reached the end of the topics in the '{section_title}' section." # Refine message
                    next_section_id = await get_next_section_id_neo4j(section_id)
                    if next_section_id:
                        # Try to get the next section's title for a better response
                        # Need a function like get_section_details(section_id) in retrieval_logic.py
                        # For now, just use the ID.
                        response_parts.append(f"The next section to explore is '{next_section_id}'.")
                    else:
                        response_parts.append("You seem to have completed the final section of this roadmap!")
                else:
                     response_parts.append("Could not determine the current section to find the next one.")

            return " ".join(response_parts)

        except Exception as e:
            logging.error(f"Error in {self.name} tool: {e}", exc_info=True)
            return f"Sorry, an error occurred while determining the next steps from topic {current_topic_id}."

    def _run(self, *args, **kwargs): raise NotImplementedError(f"{self.name} does not support sync execution.")


# --- Tool Initialization ---
# Create instances of the tools to be passed to the agent

def get_agent_tools():
    """Returns a list of tool instances for the agent."""
    if not retrieval_logic_available:
        logging.warning("Retrieval logic not available, returning empty tool list.")
        return []

    return [
        HybridRoadmapSearchTool(),
        GetTopicDetailsTool(),
        GetNextStepsTool(),
        # Add GetPreviousStepTool instance here if implemented
    ]

# --- Simple Test Function (Optional) ---
# Can be run directly: python agent_tools.py
async def test_tools_directly():
    """Requires retrieval_logic.py and DB connections to be available"""
    if not retrieval_logic_available:
        print("Cannot run tests, retrieval_logic.py failed to import.")
        return

    # Import connection management if testing standalone
    from retrieval_logic import get_neo4j_driver, close_neo4j_driver, get_chromadb_collection

    print("--- Testing Agent Tools Standalone ---")
    try:
        # Ensure connections are active for the test duration
        await get_neo4j_driver()
        get_chromadb_collection() # Initialize ChromaDB sync

        tools = get_agent_tools()
        hybrid_tool = tools[0]
        details_tool = tools[1]
        next_steps_tool = tools[2]

        print("\nTesting Hybrid Search Tool...")
        # Using .run shortcut for testing (which calls _arun via Tool implementation)
        result1 = await hybrid_tool.arun(query_text="Tell me about databases")
        print(f"Hybrid Search Result 1:\n{result1}")

        print("\nTesting Get Details Tool...")
        result3 = await details_tool.arun(topic_id="databases_relational")
        print(f"Details Result:\n{result3}")

        print("\nTesting Get Next Steps Tool...")
        result5 = await next_steps_tool.arun(current_topic_id="databases_nosql") # Should point to APIs
        print(f"Next Steps Result 1:\n{result5}")

    except Exception as e:
         print(f"An error occurred during standalone tool test: {e}")
    finally:
        await close_neo4j_driver() # Clean up Neo4j connection

    print("\n--- Agent Tool Standalone Tests Complete ---")


if __name__ == "__main__":
    # This allows testing the tools directly if needed.
    # Note: Running this directly requires Neo4j/ChromaDB to be accessible
    # and assumes retrieval_logic.py can manage its connections.
    # It's often better to test tools through the agent itself.
    # asyncio.run(test_tools_directly())
    print("Agent tools defined. Run agent_main.py to use them with an agent.")
    pass