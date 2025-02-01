#!/usr/bin/env python3
"""
Example script demonstrating how to use the iterative pipeline for multi-agent analysis.
This script shows how multiple agents collaborate to progressively refine and analyze a topic.
"""

import asyncio
from app.domain.iterative import iterative_pipeline
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

# Set up logging to see the process in action
setup_logging()

async def main():
    # Initialize the agent factory
    factory = AgentFactory()
    
    # Configure your analysis
    query = "Analyze the impact of AI on job markets"  # You can change this query
    iterations = 3  # Increase for more thorough analysis, decrease for quicker results
    
    print(f"\nStarting iterative analysis of: '{query}'")
    print(f"Running {iterations} iterations...\n")
    
    try:
        # Run the iterative pipeline
        result = await iterative_pipeline(query, iterations, factory)
        
        print("\n=== Final Analysis ===")
        print(result)
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 