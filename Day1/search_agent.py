import os
import asyncio

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

try:
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from environment variables
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

except Exception as e:
    print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your .env file. Details: {e}")

root_agent = Agent(
    name="helpful_assistant",
    model="gemini-2.0-flash-lite",
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)    
print("🤖 Root Agent defined.")

runner = InMemoryRunner(agent=root_agent)
print("Runner created.")

async def main():
    response = await runner.run_debug("What is Agent Development Kit from Google? Which all languages are supported?")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())