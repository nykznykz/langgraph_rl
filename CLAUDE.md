# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a LangGraph agent with ART (Automatic Reward Training) integration for reinforcement learning. The example demonstrates an email search agent that can be trained to improve its performance through RL using the OpenPipe ART framework.

## Architecture

The project consists of several key components:

- **`tools.py`**: Contains the agent's tools (`search_inbox`, `read_email`) with mock email data
- **`agent.py`**: Main agent implementation using LangGraph's `create_react_agent` with ART integration via `wrap_rollout` and `init_chat_model`
- **`train.py`**: Training script that runs multiple scenarios and initiates RL training using RULER reward function
- **`.env.example`**: Template for environment configuration including API keys and model settings

The agent uses ART's `TrainableModel` with Qwen/Qwen2.5-7B-Instruct as the base model, wrapped with LangGraph's ReAct pattern for tool usage.

## Common Commands

### Setup and Installation
```bash
pip install -r requirements.txt
```

### Environment Configuration
```bash
cp .env.example .env
# Edit .env with your actual API keys and configuration
```

### Run Single Agent Scenario
```bash
python agent.py           # Full ART integration (requires torch, GPU backend)
python simple_agent.py    # Simple test without ART (lightweight)
```

### Run Full Training Pipeline
```bash
python train.py
```

### Testing
Testing options in order of complexity:
1. `python simple_agent.py` - Basic LangGraph functionality test (requires OpenAI API key)
2. `python agent.py` - Full ART integration test (requires ART backend setup)
3. `python train.py` - Complete training pipeline (requires GPU + full environment)

Note: ART integration requires additional dependencies (torch, polars) and may need GPU resources for training.

## Key Implementation Details

- The agent uses `@wrap_rollout(model)` decorator to enable ART trajectory tracking
- `init_chat_model()` creates an ART-compatible chat model for the LangGraph agent
- Training scenarios cover various email search patterns (by sender, topic, content)
- Mock email database provides realistic but safe test data
- Environment variables control model selection, project naming, and training parameters

## Development Notes

- ART backend requires proper API configuration for full functionality
- The example uses `LocalBackend()` for development but can be configured for remote ART services
- Training trajectories are currently empty placeholders - real implementations would capture agent reasoning steps
- RULER reward function is used for automatic reward generation during training