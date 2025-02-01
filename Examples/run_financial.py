#!/usr/bin/env python3
"""
Example script demonstrating how to use the financial domain analysis.
This script analyzes stock data and provides investment recommendations.
"""

import asyncio
from app.domain.financial import analyze_financial
from app.agents.agent_factory import AgentFactory
from app.utils.logger import setup_logging

# Set up logging to see the process in action
setup_logging()

async def main():
    # Initialize the agent factory
    factory = AgentFactory()
    
    # Configure your analysis
    ticker = "AAPL"  # Stock symbol to analyze (e.g., AAPL for Apple)
    period = "6mo"   # Analysis period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    print(f"\nStarting financial analysis for {ticker}")
    print(f"Analysis period: {period}\n")
    
    try:
        # Run the financial analysis
        result = await analyze_financial(ticker, period, factory)
        
        print("\n=== Financial Analysis Report ===")
        print(result)
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 