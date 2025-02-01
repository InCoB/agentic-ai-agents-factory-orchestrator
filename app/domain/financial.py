# app/domain/financial.py
import logging
import yfinance as yf
from app.agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

async def analyze_financial(ticker: str, period: str, factory: AgentFactory) -> str:
    # Fetch financial data
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        logger.info("No financial data found.")
        return "No data available."
    
    # Get the latest price and calculate price changes
    latest_price = data['Close'][-1]
    price_change = ((latest_price - data['Close'][0]) / data['Close'][0]) * 100
    
    # Prepare detailed financial context
    summary_text = (
        f"Financial Analysis for {ticker}:\n"
        f"Latest Price: ${latest_price:.2f}\n"
        f"Price Change: {price_change:.2f}%\n"
        f"Statistical Summary:\n{data.describe().to_string()}"
    )
    
    # Get initial summary from SummarizerAgent
    summary_result = await factory.get_agent("SummarizerAgent").process(
        f"Analyze this financial data and provide key insights:\n{summary_text}"
    )
    
    # Pass the summary to DecisionAgent for investment advice
    decision_prompt = (
        f"Based on the following analysis of {ticker} stock:\n\n"
        f"{summary_result}\n\n"
        f"Current Price: ${latest_price:.2f}\n"
        f"Price Change: {price_change:.2f}%\n\n"
        "Provide specific investment recommendations. Consider:\n"
        "1. Current price trends\n"
        "2. Historical volatility\n"
        "3. Trading volume patterns\n"
        "4. Potential risks and opportunities"
    )
    
    decision_result = await factory.get_agent("DecisionAgent").process(decision_prompt)
    
    # Get a final review from ValidatorAgent
    validation_prompt = (
        f"Review this investment analysis for {ticker}:\n\n"
        f"Summary:\n{summary_result}\n\n"
        f"Recommendations:\n{decision_result}\n\n"
        "Validate the analysis and add any missing critical points."
    )
    validation_result = await factory.get_agent("ValidatorAgent").process(validation_prompt)
    
    final_report = "\n\n".join([
        f"=== Financial Analysis Report for {ticker} ===",
        f"Period: {period}",
        f"Current Price: ${latest_price:.2f} ({price_change:+.2f}%)",
        "\n=== Market Analysis ===",
        summary_result,
        "\n=== Investment Recommendations ===",
        decision_result,
        "\n=== Validation & Additional Insights ===",
        validation_result
    ])
    
    logger.info(f"Financial analysis completed for {ticker}")
    return final_report
