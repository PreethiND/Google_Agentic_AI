import os
import asyncio

from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
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

# Research Agent: Its job is to use the google_search tool and present findings.
research_agent = Agent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    tools=[google_search],
    output_key="research_findings", # The result of this agent will be stored in the session state with this key.
)

print("✅ research_agent created.")

runner = InMemoryRunner(agent=research_agent)

# Summarizer Agent: Its job is to summarize the text it receives.
summarizer_agent = Agent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided research findings: {research_findings}
Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary",
)

print("✅ summarizer_agent created.")

# Outline Agent: Creates the initial blog post outline.
outline_agent = Agent(
    name="OutlineAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Create a blog outline for the given topic with:
    1. A catchy headline
    2. An introduction hook
    3. 3-5 main sections with 2-3 bullet points for each
    4. A concluding thought""",
    output_key="blog_outline", # The result of this agent will be stored in the session state with this key.
)

print("✅ outline_agent created.")

# Writer Agent: Writes the full blog post based on the outline from the previous agent.
writer_agent = Agent(
    name="WriterAgent",
    model="gemini-2.5-flash-lite",
    # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
    instruction="""Following this outline strictly: {blog_outline}
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
    output_key="blog_draft", # The result of this agent will be stored with this key.
)

print("✅ writer_agent created.")

# Editor Agent: Edits and polishes the draft from the writer agent.
editor_agent = Agent(
    name="EditorAgent",
    model="gemini-2.5-flash-lite",
    # This agent receives the `{blog_draft}` from the writer agent's output.
    instruction="""Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
    output_key="final_blog", # This is the final output of the entire pipeline.
)

print("✅ editor_agent created.")

root_agent = SequentialAgent(
    name="BlogPostPipeline",
    sub_agents=[
        outline_agent,
        writer_agent,
        editor_agent
    ],
)
print("✅ Sequential Agent created.")

async def main():
    response = await runner.run_debug("Write a blog post about the benefits of multi-agent systems for software developers")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())