# LangGraph Reinforcement Learning Agent

A demonstration of LangGraph agents enhanced with reinforcement learning using ART (Automatic Reward Training) from OpenPipe. This project implements an email search agent that can be trained to improve its performance through RL.

## Overview

This project showcases how to:
- Create a LangGraph ReAct agent with custom tools
- Integrate ART for reinforcement learning training
- Use RULER reward function for automatic reward generation
- Train agents on multiple scenarios to improve performance

## Features

- **Email Search Agent**: Mock email search and retrieval functionality
- **ART Integration**: Automatic trajectory tracking and reward-based training
- **ReAct Pattern**: Uses LangGraph's reasoning and acting framework
- **Configurable Training**: Environment-based configuration for models and training parameters

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your ART API key and other configuration
```

### 3. Run Single Agent Test
```bash
python agent.py
```

### 4. Run Full Training Pipeline  
```bash
python train.py
```

## Project Structure

- `tools.py` - Agent tools for email search and retrieval
- `agent.py` - Main agent implementation with ART integration
- `train.py` - Training script with multiple scenarios
- `.env.example` - Environment configuration template
- `CLAUDE.md` - Detailed development guide

## Training Process

The agent is trained on scenarios like:
- Finding emails from specific senders
- Searching by topic or content
- Retrieving latest messages
- Budget and finance-related queries

Training uses ART's RULER reward function to automatically generate rewards based on agent performance.

## Requirements

- Python 3.8+
- ART (Automatic Reward Training)
- LangGraph
- LangChain

See `requirements.txt` for complete dependencies.
