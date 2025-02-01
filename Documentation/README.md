# AI Agent Factory Documentation

Agentic AI Agents Factory Orchestrator -is modular, and asynchronous AI agent factory designed for AI-made dynamic workflow orchestration using LLM integration. Build scalable, agentic automation pipelines with robust error handling, plugin-based capabilities, and high-performance concurrent operations or outsource workflow building to Operator Agent.

## Table of Contents

## Quick Start Guide

### Installation

1. **Clone the Repository and Setup:**
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your API keys
  
  

### Running the Application

1. **Main Application (`main.py`)**
   ```bash
   # Start the FastAPI server
   uvicorn main:app --reload --port 8000
   ```
   This launches the REST API server with endpoints for agent operations.

2. **Gradio Interface (`gradio_app.py`)**
   ```bash
   # Start the Gradio web interface
   python gradio_app.py
   ```
   Access the interface at http://127.0.0.1:7860 in your browser.

### Running Examples

The project includes several examples demonstrating different use cases:

1. **Basic Operator Examples:**
```bash
# From the project root
python -m Examples.example_operator_usage           # Basic operator usage
python -m Examples.example_operator_usage_fixed     # Fixed workflow example
python -m Examples.example_operator_usage_dynamic   #  Operator creates a workflow and prompts while agents execute it
```

2. **Iterative Pipeline Example:**
Create a new file `Examples/run_iterative.py`:
```python
[example code remains the same...]
```

Run it using:
```bash
# From the project root
python -m Examples.run_iterative
```

3. **Domain-Specific Examples:**
You can create custom scripts for specific domains in the `app/domain/` directory. Each domain can have its own specialized workflow and agent interactions.

Example structure for a new domain:
```python
# app/domain/custom_domain.py
import asyncio
from app.agents.agent_factory import AgentFactory

async def custom_domain_pipeline(input_data: str, factory: AgentFactory) -> str:
    # Get required agents
    data_agent = factory.get_agent("DataAgent")
    creative_agent = factory.get_agent("CreativeAgent")
    
    # Process through domain-specific workflow
    data_result = await data_agent.process(input_data)
    creative_result = await creative_agent.process(data_result)
    
    return f"Analysis Results:\n{creative_result}"
```

Run it with a script:
```python
# Examples/run_custom_domain.py
import asyncio
from app.domain.custom_domain import custom_domain_pipeline
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

setup_logging()

async def main():
    factory = AgentFactory()
    result = await custom_domain_pipeline("Your input here", factory)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

Available domains in `app/domain/`:
- `iterative.py`: Multi-agent iterative analysis
- `financial.py`: Financial data analysis
- `news.py`: News content analysis
- `local_file.py`: Local file processing

Each domain can be run using a similar pattern of creating a runner script in the Examples directory.

## Documentation Structure

### Core Documentation
- [`Documentation.md`](./Documentation.md): Complete technical documentation
  - General Overview & Architecture
  - Detailed Structure & Component Analysis
  - Integration & Workflows
  - API Reference & Advanced Usage

### Interface Documentation
- [`gradio_interface.md`](./gradio_interface.md): Detailed documentation of the Gradio web interface
  - Features and components
  - Usage instructions
  - Technical details
  - Troubleshooting guide

## Documentation Contents

### Part 1: General Overview & Architecture
- Introduction and purpose
- Key features and system requirements
- Core architecture and design philosophy
- Project structure and configuration

### Part 2: Detailed Structure & Component Analysis
- App module analysis
- Agent system details
- Services layer implementation
- Metrics and utilities

### Part 3: Integration & Workflows
- Agent interaction patterns
- Custom agent development
- Workflow creation & orchestration
- Error handling strategies
- Performance optimization
- Security considerations

### Part 4: API Reference & Advanced Usage
- API endpoints reference
- Agent factory methods
- Advanced customization options
- Best practices & troubleshooting

## Using the Documentation

1. Start with `Documentation.md` for technical understanding
2. Refer to `gradio_interface.md` for web interface usage
3. Each document uses Markdown formatting for better readability
4. Code examples and configurations are properly formatted
5. Screenshots and diagrams (if any) are stored in the `assets` folder

## Contributing to Documentation

When adding or updating documentation:
1. Follow the existing Markdown formatting
2. Include clear examples where appropriate
3. Update the README.md when adding new documents
4. Maintain consistent style and structure

## Getting Help

If you find any issues or need clarification:
1. Check the troubleshooting sections
2. Review the detailed logs
3. Refer to the example use cases
4. Contact the development team for support

