import os
import logging
import asyncio
from dotenv import load_dotenv
import pprint # For testing output

# LLM Imports (Example using OpenAI)
from langchain_openai import ChatOpenAI

# Memory Imports
from mem0 import AsyncMemory # Use AsyncMemory

# LangChain Imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage # For formatting chat history

# Custom Tools Import
try:
    from agent_tools import get_agent_tools # Import the function that returns tool instances
    tools_available = True
except ImportError:
    logging.error("Could not import get_agent_tools from agent_tools. Ensure agent_tools.py exists and is accessible.")
    tools_available = False


# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Initialization ---
llm = None
try:
    # Example using OpenAI GPT-4o mini (adjust model name as needed)
    # Using a model that supports function calling is recommended for create_openai_functions_agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    logging.info(f"Initialized LLM: {llm.model_name}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}", exc_info=True)

# --- Mem0 Initialization ---
agent_memory = None
try:
    agent_memory = AsyncMemory() # Use AsyncMemory()
    logging.info("Initialized AsyncMemory.")
except Exception as e:
    logging.error(f"Failed to initialize AsyncMemory: {e}", exc_info=True)


# --- Agent Setup ---
agent_executor = None
if llm and tools_available and agent_memory:
    logging.info("Setting up LangChain Agent...")
    try:
        # 1. Get Tools
        tools = get_agent_tools()
        if not tools:
             logging.warning("No tools loaded. Agent functionality will be limited.")

        # 2. Define Agent Prompt
        # Instructs the agent on its role, how to use tools, and conversational context
        # MEMORY_KEY and AGENT_SCRATCHPAD are special placeholders
        MEMORY_KEY = "chat_history"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant guiding users through the Backend Developer roadmap. "
                    "Your goal is to answer questions about topics, suggest next steps, and provide details using the available tools. "
                    "Be concise and clear in your responses. Use the tools provided when necessary to answer questions about the roadmap content, structure, or sequence. "
                    "If asked about something unrelated to the roadmap, politely state that you can only help with the backend learning path. "
                    "Use the chat history to understand the context of the conversation."
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY), # Where chat history goes
                ("user", "{input}"), # User's current message
                MessagesPlaceholder(variable_name="agent_scratchpad"), # Where agent intermediate steps go
            ]
        )

        # 3. Create Agent
        # Using OpenAI Functions agent - requires an OpenAI model that supports function calling
        agent = create_openai_functions_agent(llm, tools, prompt)

        # 4. Create Agent Executor
        # This runs the agent loop (thought -> action -> observation -> thought...)
        agent_executor = AgentExecutor(
            agent=agent,
                tools=tools,
                verbose=True, # Set to True to see agent's thought process, tool calls, etc.
                handle_parsing_errors=True, # Try to gracefully handle LLM output parsing errors
                # We will manually manage Mem0 history input/output for now
                # memory=... # LangChain memory object could be added here if Mem0 had direct integration
            )
        logging.info("LangChain Agent Executor created successfully.")

    except Exception as e:
        logging.error(f"Failed to create LangChain Agent Executor: {e}", exc_info=True)
else:
    logging.error("Agent Executor cannot be created due to missing LLM, Tools, or Memory.")


# --- Updated Agent Interaction Function ---
async def run_agent_interaction(user_id: str, message: str):
    """Runs the full agent interaction using LangChain AgentExecutor and Mem0."""
    if not agent_executor:
        return "Error: Agent Executor not initialized."
    if not agent_memory:
        return "Error: Memory system not initialized."

    logging.info(f"Running agent interaction for user '{user_id}': '{message}'")

    try:
        # 1. Retrieve relevant conversational history from Mem0
        # Limit the number of messages to avoid excessive context
        logging.debug(f"Searching Mem0 for history relevant to: '{message}' for user: {user_id}")
        # Note: Mem0 search might return summaries or raw messages depending on config/version
        # We might need to format the results appropriately for LangChain prompt
        raw_memories = await agent_memory.search(query=message, user_id=user_id, limit=5) # Get recent/relevant memories
        logging.debug(f"Raw memories retrieved: {raw_memories}")

        # Format history for LangChain MessagesPlaceholder (simple approach for now)
        # Assumes search results give text content - adjust based on actual Mem0 output
        chat_history_messages = []
        # This formatting is basic - might need adjustment based on Mem0 search results structure
        if raw_memories and 'results' in raw_memories:
             for mem in reversed(raw_memories['results']): # Process in chronological order
                mem_text = mem.get('text', '')
                # Basic role detection (improve if Mem0 provides role metadata)
                if mem_text.lower().startswith("agent:"):
                    chat_history_messages.append(AIMessage(content=mem_text.replace("Agent: ","", 1)))
                else:
                    # Assume user if not agent (refine this logic)
                    chat_history_messages.append(HumanMessage(content=mem_text))


        logging.debug(f"Formatted chat history for agent: {chat_history_messages}")

        # 2. Invoke the LangChain Agent Executor
        # Pass the user input and the formatted chat history
        response = await agent_executor.ainvoke({
            "input": message,
            "chat_history": chat_history_messages
        })

        agent_response = response.get("output", "Sorry, I encountered an issue.")

        # 3. Add current user message and agent response to Mem0
        # Use user_id to scope memory
        await agent_memory.add(message, user_id=user_id) # Storing simple text, role might be implicit
        # Add agent response (maybe prefix for clarity if role isn't stored)
        await agent_memory.add(f"Agent: {agent_response}", user_id=user_id) # Prefixing response
        logging.info("Added user message and agent response to AsyncMemory.")

        # 4. Return agent's final response
        return agent_response

    except Exception as e:
        logging.error(f"Error during agent interaction: {e}", exc_info=True)
        return "Sorry, an error occurred while processing your request."


async def main_test():
    """Test basic setup and agent interaction."""
    logging.info("--- Starting Phase 3 Agent Tests ---")
    if agent_executor and agent_memory:
        test_user = "user_dev_1" # Use a distinct user ID for testing

        # Test conversation flow
        messages = [
            "Hi there!",
            "What is REST?",
            "What should I learn after that?", # Should use the previous context
            "Tell me more about 'Databases'",
            "What follows the testing section?",
            "Give me details about topic 'ci_cd_concepts'"
        ]

        for msg in messages:
            print("\n-----------------------------")
            print(f"User ({test_user}): {msg}")
            response = await run_agent_interaction(test_user, msg)
            print(f"Agent: {response}")
            await asyncio.sleep(1) # Small delay between interactions if needed

    else:
        logging.error("Skipping interaction test due to initialization errors.")
    logging.info("--- Phase 3 Agent Tests Finished ---")


if __name__ == "__main__":
    # Run test interaction
    # Note: Need to properly close Neo4j driver if tests use it directly
    # The agent tools call retrieval_logic, which manages the driver globally for now.
    # Ensure cleanup happens if the script runs long or has many interactions.
    try:
        asyncio.run(main_test())
    finally:
        # Ensure Neo4j driver is closed on script exit
        # Need to import and call close_neo4j_driver from retrieval_logic
        # This assumes retrieval_logic exposes a cleanup function or manages it internally
        # For simplicity now, we rely on the script ending. In a server, explicit cleanup is vital.
        # Example (if close function exists in retrieval_logic):
        # from retrieval_logic import close_neo4j_driver as close_retrieval_driver
        # asyncio.run(close_retrieval_driver())
        logging.info("Main script finished.")