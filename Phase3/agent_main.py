import os
import logging
import asyncio
from dotenv import load_dotenv
import pprint # Keep for potential debugging

# LLM Imports
from langchain_openai import ChatOpenAI

# Memory Imports
from mem0 import AsyncMemory

# LangChain Imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Custom Tools Import
try:
    from agent_tools import get_agent_tools
    tools_available = True
except ImportError:
    logging.error("CRITICAL: Could not import get_agent_tools from agent_tools. Ensure agent_tools.py exists and is accessible.")
    tools_available = False

# === NEW IMPORTS for API ===
import uvicorn # ASGI server
from fastapi import FastAPI
from pydantic import BaseModel
# === END NEW IMPORTS ===


# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Initialization ---
llm = None
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    logging.info(f"Initialized LLM: {llm.model_name}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}", exc_info=True)

# --- Mem0 Initialization ---
agent_memory = None
try:
    agent_memory = AsyncMemory()
    logging.info("Initialized AsyncMemory.")
except Exception as e:
    logging.error(f"Failed to initialize AsyncMemory: {e}", exc_info=True)

# --- Agent Setup ---
agent_executor = None
if llm and tools_available and agent_memory:
    logging.info("Setting up LangChain Agent...")
    try:
        tools = get_agent_tools()
        if not tools: logging.warning("No tools loaded.")

        MEMORY_KEY = "chat_history"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant guiding users through the Backend Developer roadmap..."), # Keep your detailed prompt
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        logging.info("LangChain Agent Executor created successfully.")
    except Exception as e:
        logging.error(f"Failed to create LangChain Agent Executor: {e}", exc_info=True)
else:
    logging.error("Agent Executor cannot be created due to missing LLM, Tools, or Memory.")


# --- Core Agent Interaction Logic ---
# (run_agent_interaction function remains the same as before)
async def run_agent_interaction(user_id: str, message: str):
    """Runs the full agent interaction using LangChain AgentExecutor and Mem0."""
    if not agent_executor: return "Error: Agent Executor not initialized."
    if not agent_memory: return "Error: Memory system not initialized."
    logging.info(f"Running agent interaction for user '{user_id}': '{message}'")
    try:
        # 1. Retrieve relevant conversational history from Mem0
        logging.debug(f"Searching Mem0 for history relevant to: '{message}' for user: {user_id}")
        raw_memories = await agent_memory.search(query=message, user_id=user_id, limit=5)
        logging.debug(f"Raw memories retrieved: {raw_memories}")
        chat_history_messages = []
        if raw_memories and 'results' in raw_memories:
             for mem in reversed(raw_memories['results']):
                mem_text = mem.get('text', '')
                # Basic role detection (improve if Mem0 provides role metadata)
                if mem_text.lower().startswith("agent:"):
                    chat_history_messages.append(AIMessage(content=mem_text.replace("Agent: ","", 1)))
                else:
                    chat_history_messages.append(HumanMessage(content=mem_text))
        logging.debug(f"Formatted chat history for agent: {chat_history_messages}")

        # 2. Invoke the LangChain Agent Executor
        response = await agent_executor.ainvoke({
            "input": message,
            "chat_history": chat_history_messages
        })
        agent_response = response.get("output", "Sorry, I encountered an issue.")

        # 3. Add current user message and agent response to Mem0
        await agent_memory.add(message, user_id=user_id)
        await agent_memory.add(f"Agent: {agent_response}", user_id=user_id)
        logging.info("Added user message and agent response to AsyncMemory.")

        # 4. Return agent's final response
        return agent_response
    except Exception as e:
        logging.error(f"Error during agent interaction: {e}", exc_info=True)
        return "Sorry, an error occurred while processing your request."


# === NEW FastAPI Setup ===

# Define request and response models using Pydantic
class ChatMessage(BaseModel):
    user_id: str = "default_user" # Default user ID if none provided
    message: str

class AgentResponse(BaseModel):
    response: str

# Create FastAPI app instance
# Add title and description for API docs
app = FastAPI(
    title="Personalized Learning Path Navigator API",
    description="API endpoint for interacting with the backend roadmap agent.",
    version="0.1.0",
)

@app.post("/chat", response_model=AgentResponse)
async def chat_endpoint(request: ChatMessage):
    """
    Receive a user message and get a response from the learning path agent.
    """
    logging.info(f"Received chat request for user '{request.user_id}'")
    agent_reply = await run_agent_interaction(request.user_id, request.message)
    return AgentResponse(response=agent_reply)

# Add a simple root endpoint for testing server status
@app.get("/")
async def root():
    return {"message": "Learning Path Navigator Agent API is running."}

# === END FastAPI Setup ===


# --- Main execution block (runs the server) ---
if __name__ == "__main__":
    # Remove or comment out the previous main_test() call
    # asyncio.run(main_test())

    # Use uvicorn to run the FastAPI app
    # You can configure host and port as needed
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Note: Proper cleanup of resources like the Neo4j driver on server shutdown
    # would typically be handled using FastAPI's lifespan events, but is omitted
    # here for simplicity in this development script.