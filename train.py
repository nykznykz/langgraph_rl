"""Training script for the LangGraph email agent using ART."""

import asyncio
import art
from art.langgraph import wrap_rollout, init_chat_model
from art.local import LocalBackend
from langgraph.prebuilt import create_react_agent
from tools import search_inbox, read_email
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define tools for the agent
tools = [search_inbox, read_email]

# Training scenarios for the email agent
TRAINING_SCENARIOS = [
    "Find emails from John about the quarterly report",
    "Search for emails containing budget discussions", 
    "Find the latest email from Sarah",
    "Look for emails about project updates",
    "Search for finance-related emails",
    "Find emails from the finance team about budget approval",
    "Look for John's follow-up emails",
    "Search for Sarah's weekly summaries",
    "Find emails with quarterly numbers",
    "Look for budget-related correspondence"
]


async def train_email_agent():
    """Main training function for the email agent."""
    print("Starting ART training for LangGraph email agent...")
    
    with LocalBackend() as backend:
        # Create trainable model with environment configuration
        model_name = os.getenv("MODEL_NAME", "email-agent-langgraph")
        project_name = os.getenv("PROJECT_NAME", "email-search-agent")
        base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        
        model = art.TrainableModel(
            name=model_name,
            project=project_name,
            base_model=base_model,
        )

        print(f"Registering model: {model_name}")
        await backend.register(model)

        # Define rollout function with ART integration
        async def run_agent(scenario: str) -> art.Trajectory:
            """Run the agent on a scenario and return trajectory for training."""
            print(f"Running scenario: {scenario}")
            
            # Create the ReAct agent with ART-integrated chat model
            agent = create_react_agent(init_chat_model(), tools)
            
            # Run the agent on the scenario
            try:
                result = await agent.ainvoke({"messages": [("user", scenario)]})
                print(f"Agent completed scenario successfully")
                
                # In a real implementation, you would extract meaningful trajectory data
                # For now, we return an empty trajectory as per the example
                return art.Trajectory()
                
            except Exception as e:
                print(f"Error running scenario: {e}")
                return art.Trajectory()

        # Apply the wrap_rollout decorator
        wrapped_agent = wrap_rollout(model, run_agent)

        # Run training scenarios
        print(f"Running {len(TRAINING_SCENARIOS)} training scenarios...")
        
        for i, scenario in enumerate(TRAINING_SCENARIOS, 1):
            print(f"\n--- Scenario {i}/{len(TRAINING_SCENARIOS)} ---")
            await wrapped_agent(scenario)
        
        # Start training with RULER reward function
        reward_function = os.getenv("REWARD_FUNCTION", "ruler")
        print(f"\nStarting training with reward function: {reward_function}")
        
        try:
            await art.train(model, reward_function=reward_function)
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Training error: {e}")
            print("This might be expected if ART backend is not fully configured")


async def run_evaluation_scenarios():
    """Run evaluation scenarios to test the trained agent."""
    print("\n=== Running Evaluation Scenarios ===")
    
    eval_scenarios = [
        "Find all emails from John",
        "What budget emails do we have?", 
        "Show me Sarah's most recent email"
    ]
    
    with LocalBackend() as backend:
        model = art.TrainableModel(
            name=os.getenv("MODEL_NAME", "email-agent-langgraph"),
            project=os.getenv("PROJECT_NAME", "email-search-agent"),
            base_model=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )

        await backend.register(model)

        async def run_agent(scenario: str) -> art.Trajectory:
            agent = create_react_agent(init_chat_model(), tools)
            result = await agent.ainvoke({"messages": [("user", scenario)]})
            return art.Trajectory()

        wrapped_agent = wrap_rollout(model, run_agent)

        for scenario in eval_scenarios:
            print(f"\nEvaluating: {scenario}")
            await wrapped_agent(scenario)


if __name__ == "__main__":
    print("LangGraph + ART Email Agent Training")
    print("====================================")
    
    # Run training
    asyncio.run(train_email_agent())
    
    # Optional: Run evaluation
    run_eval = input("\nRun evaluation scenarios? (y/n): ").lower().startswith('y')
    if run_eval:
        asyncio.run(run_evaluation_scenarios())
    
    print("\nTraining session completed!")