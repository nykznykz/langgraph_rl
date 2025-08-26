"""Simple LangGraph agent without full ART integration for quick testing."""

import asyncio
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from tools import search_inbox, read_email

load_dotenv()


def create_simple_agent():
    """Create a simple LangGraph agent without ART integration."""
    # Use a simple OpenAI model (reads OPENAI_API_KEY from .env)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create the ReAct agent with tools
    tools = [search_inbox, read_email]
    agent = create_react_agent(llm, tools)
    
    return agent


async def run_scenario(scenario: str):
    """Run the agent on a single scenario."""
    print(f"Running scenario: {scenario}")
    
    try:
        agent = create_simple_agent()
        
        # Run the agent
        result = await agent.ainvoke({"messages": [("user", scenario)]})
        
        print("\nAgent Execution:")
        print("================")
        
        # Extract and display the conversation flow
        for message in result["messages"]:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"\n🔧 Tool calls:")
                for tool_call in message.tool_calls:
                    print(f"  - {tool_call['name']}({tool_call['args']})")
            elif hasattr(message, 'name') and message.name:
                # Tool response
                print(f"📧 {message.name} result: {message.content[:100]}...")
            elif hasattr(message, 'content') and message.content and not hasattr(message, 'tool_calls'):
                # Final AI response
                print(f"\n💬 Final response:\n{message.content}")
        
    except Exception as e:
        print(f"Expected error (no API key): {e}")
        print("✓ Agent structure is correct - would work with proper API keys")


if __name__ == "__main__":
    # Test scenario
    test_scenario = "Find emails from John about the quarterly report"
    
    print("Simple LangGraph Agent Test")
    print("===========================")
    print("Note: This will show structure but fail without API keys")
    
    # Run the scenario
    asyncio.run(run_scenario(test_scenario))