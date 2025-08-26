"""LangGraph agent with ART integration for reinforcement learning."""

import art
import os
from art.langgraph import wrap_rollout, init_chat_model
from art.local import LocalBackend
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from tools import search_inbox, read_email
from dotenv import load_dotenv

load_dotenv()


# Define tools for the agent
tools = [search_inbox, read_email]


async def create_email_agent():
    """Create and return a configured email agent with ART integration."""
    with LocalBackend() as backend:
        # Create trainable model
        model = art.TrainableModel(
            name="email-agent-langgraph",
            project="email-search-agent", 
            base_model="Qwen/Qwen2.5-7B-Instruct",
        )

        await backend.register(model)

        # Define rollout function with ART integration
        async def run_agent(scenario: str) -> art.Trajectory:
            """Run the agent on a given scenario and return trajectory."""
            # Create the ReAct agent with ART-integrated chat model
            # Try using environment variable for model or fallback to direct ChatOpenAI
            try:
                llm = init_chat_model(model=os.getenv("BASE_MODEL", "gpt-3.5-turbo"))
            except Exception:
                # Fallback to direct ChatOpenAI if init_chat_model fails
                llm = ChatOpenAI(
                    model=os.getenv("BASE_MODEL", "gpt-3.5-turbo"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.1
                )
            agent = create_react_agent(llm, tools)
            
            # Run the agent on the scenario
            result = await agent.ainvoke({"messages": [("user", scenario)]})
            
            # Return trajectory for ART training (placeholder implementation)
            return art.Trajectory(
                messages_and_choices=[],  # Would contain conversation messages
                reward=0.0  # Would contain actual reward from training
            )

        # Apply the wrap_rollout decorator
        wrapped_agent = wrap_rollout(model, run_agent)
        
        return wrapped_agent, model


async def run_single_scenario(scenario: str):
    """Run the agent on a single scenario."""
    agent_func, model = await create_email_agent()
    result = await agent_func(scenario)
    print(f"Completed scenario: {scenario}")
    return result


if __name__ == "__main__":
    import asyncio
    
    # Example scenario
    test_scenario = "Find emails from John about the quarterly report"
    
    # Run the scenario
    asyncio.run(run_single_scenario(test_scenario))