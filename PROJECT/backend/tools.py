from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from datetime import date
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(api_key=os.getenv("GROQ_KEY"), model='meta-llama/llama-4-scout-17b-16e-instruct')

chat_histories = {}

def get_memory(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

@tool
def company_information(ticker: str)-> dict:
    """ Use this tool to retrieve company information like the residential address, industry,
    sector, company officers, business summary, website links, marketCap, current price, ebitda, total debt, total revenue, debt-to-equity, and other such
    financial information about the company. """
    ticker_obj = yf.Ticker(ticker)
    ticker_info = ticker_obj.get_info()
    return ticker_info

@tool
def historical_price_data(ticker: str, period: str = "1y", interval: str = "1d") -> dict:
    """
    Retrieves historical market data for the specified ticker.
    Parameters:
        ticker (str): Stock ticker symbol.
        period (str): Data period (e.g., '1d', '5d', '1mo', '1y', '5y', 'max').
        interval (str): Data interval (e.g., '1m', '5m', '1h', '1d', '1wk', '1mo').
    Returns:
        dict: Historical market data.
    """
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period, interval=interval)
    return hist.reset_index().to_dict(orient='records')

@tool
def analyst_recommendations(ticker: str) -> dict:
    """
    Retrieves analyst recommendations for the specified ticker.
    Parameters:
        ticker (str): Stock ticker symbol.
    Returns:
        dict: Analyst recommendations data.
    """
    ticker_obj = yf.Ticker(ticker)
    recs = ticker_obj.recommendations
    return recs.reset_index().to_dict(orient='records')

@tool
def earnings_calendar(ticker: str) -> dict:
    """
    Retrieves upcoming earnings dates for the specified ticker.
    Parameters:
        ticker (str): Stock ticker symbol.
    Returns:
        dict: Upcoming earnings dates.
    """
    ticker_obj = yf.Ticker(ticker)
    calendar_df = ticker_obj.calendar
    return calendar_df

@tool
def financial_statements(ticker: str) -> dict:
    """
    Retrieves financial statements for the specified ticker.
    Parameters:
        ticker (str): Stock ticker symbol.
    Returns:
        dict: Financial statements including income statement, balance sheet, and cash flow.
    """
    ticker_obj = yf.Ticker(ticker)
    return {
        "income_statement": ticker_obj.financials.to_dict(),
        "balance_sheet": ticker_obj.balance_sheet.to_dict(),
        "cash_flow": ticker_obj.cashflow.to_dict()
    }

@tool
def company_last_dividend_and_earning_date(ticker: str) -> dict:
    """ Use this tool to retrieve the company's top mutual fund holders. 
    Also this tool must return the mutual fund holder's percentage of share, stock count, and value of holdings in the company"""
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj.get_calendar()

@tool
def summary_of_mutual_fund_holders(ticker: str) -> dict:
    """ Use this tool to retrieve company's top mutual fund holders. 
    It also returns the mutual fund holder's percentage of share, stock count and value of holdings."""
    ticker_obj = yf.Ticker(ticker)
    mf_holders = ticker_obj.get_mutualfund_holders()
    return mf_holders.to_dict(orient='records')


@tool
def summary_of_institutional_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top institutional holders. 
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)   
    inst_holders = ticker_obj.get_institutional_holders()
    
    return inst_holders.to_dict(orient="records")


@tool
def stock_grade_upgrades_downgrades(ticker: str) -> dict:
    """ Use this tool to retrieve grade ratings upgrades and downgrade details of the particular stock.
    This tool must provide the name of the firms along with 'To Grade' and 'From Grade' details. 
    Grade date is also provided in this tool. """
    ticker_obj = yf.Ticker(ticker)
    curr_year = date.today().year
    upgrades_downgrades = ticker_obj.get_upgrades_downgrades()
    upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index > f"{curr_year}-01-01"]
    upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["Action"].isin(["up", "down"])]

    return upgrades_downgrades.to_dict(orient='records')

@tool
def stock_split_history(ticker: str) -> dict:
    """ Use this tool to retrieve the companies historical stock splits data. """
    ticker_obj = yf.Ticker(ticker)
    hist_stock_splits = ticker_obj.get_splits()
    return hist_stock_splits.to_dict()

@tool
def stock_news(ticker: str) -> dict:
    """Use this to retrieve latest news articles discussing particular stock ticker."""
    ticker_obj = yf.Ticker(ticker)
    
    return ticker_obj.get_news()


tools = [
    company_information,
    company_last_dividend_and_earning_date,
    stock_split_history,
    summary_of_mutual_fund_holders,
    summary_of_institutional_holders,
    stock_grade_upgrades_downgrades,
    stock_news,
    analyst_recommendations,
    historical_price_data,
    earnings_calendar,
    financial_statements
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'You are a helpful assistant. Try to answer user query using available tools.'
        ),
        MessagesPlaceholder(variable_name='messages'),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ]
)
finance_agent = create_tool_calling_agent(llm, tools, prompt)
finance_agent_executor = AgentExecutor(agent=finance_agent, tools=tools, verbose=True)

finance_agent_executor_history = RunnableWithMessageHistory(
    finance_agent_executor,
    get_memory,
    input_messages_key='messages',
    history_messages_key='chat_history',
)

def agent_response(query):
    ag_response = finance_agent_executor_history.invoke({'messages':[HumanMessage(content=query)]}, 
                                        config={"configurable": {"session_id": "test-session-1"}})
    
    return ag_response['output']
    

    
