import uvicorn
import requests
import os
import google.generativeai as genai
from serpapi.google_search import GoogleSearch
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List 
from datetime import datetime
from app import rag_pipeline, SemanticCache # Assuming 'app' refers to a local package or module
from dotenv import load_dotenv
from crawler import crawl_parallel

FINANCE_PROMPT = """
You are an AI Research Assistant specializing in financial markets (stocks, cryptocurrencies, economic trends), but capable of researching any topic. 
Assume queries about 'the market,' 'stocks,' etc., refer to financial markets unless specified. 
Your task is to provide a detailed, narrative report using max up to 3 targeted web searches, embodying a DeepSearch approach.

[Todays Date is {todays_date}]

1.  **Analyze Query & Strategize Search:** Understand user intent. If technical concepts (e.g., moving averages, algorithms) are involved, prepare to explain them thoroughly. 
    Critically plan your 3 web searches to maximize relevant information extraction for the report.
2.  **Extract & Synthesize Information:** From search results, pull key details: definitions, explanations, examples of applications (especially in finance), comparisons between methods/tools, and any limitations. Synthesize findings from all searches.
3.  **Expand with Context & Knowledge:** If search results are limited, use your general knowledge to provide a comprehensive overview. 
    Include practical examples, compare traditional vs. modern approaches (e.g., moving averages vs. machine learning), and discuss effectiveness, reliability, and limitations.
4.  **Structure the Report (Markdown, ~500+ words):**
    *   Write a detailed, narrative response in markdown format (headings, bullets, bold). **Do NOT return JSON.**
    *   **Introduction:** Briefly introduce the topic and its relevance.
    *   **Detailed Explanation:** Explain the core concept(s) in depth, including mechanics and applications.
    *   **Comparative Analysis / Further Applications:** (As applicable) Compare with other methods or provide more examples.
    *   **Limitations and Challenges:** Discuss drawbacks.
    *   **Conclusion:** Summarize key points.
    *   *Adapt section titles and content for non-financial topics as appropriate.*
5.  **Handle Missing Data:** If specific data is unavailable from searches, note it explicitly and focus on explaining concepts and their applications.

Ensure the response is engaging, informative, and suitable for users seeking a deep understanding of the subject.
"""

load_dotenv()
if not os.getenv("GOOGLE_API_MODEL"): 
    raise ValueError("GOOGLE_API_MODEL not found in .env")
if not os.getenv("SERPAPI_API_KEY"):
    raise ValueError("SERPAPI_API_KEY not found in .env")

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_MODEL"))
except Exception as e:
    print(f"Error configuring Gemini API: {str(e)}")
    raise
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class Message(BaseModel):
    role: str
    content: str

# Updated Pydantic model for request to include history
class SearchRequest(BaseModel):
    query: str
    history: List[Message] = []

data_from_files = 'crawled_data'
rag_chain = rag_pipeline(data_from_files)
cache = SemanticCache()

def web_search(query: str) -> str:
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY")
        })
        result = search.get_dict()
        organic_results = result.get('organic_results', [])
        snippets = [item['snippet'] for item in organic_results[:5]]
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        print(f"SerpAPI error: {str(e)}")
        return f"Search failed: {str(e)}"

# Function declaration for Gemini (unchanged)
web_search_declaration = {
    "name": "web_search",
    "description": "Perform a web search and return snippets from top results",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    },
}

# Initialize Gemini model 
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    raise

todays_date = datetime.today().strftime("%d-%m-%Y")

@app.post("/deepsearch")
async def deep_search(request: SearchRequest):
    try:
        print(f"Received query: {request.query}")
        
        # Convert history to Gemini-compatible format
        history = [
            {"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]}
            for msg in request.history
        ]

        # Start a chat session with history
        chat = model.start_chat(history=history)

        # Preprocess query for finance intent
        query = request.query.lower()
        refined_query = query
        if "market" in query or "stocks" in query:
            if "outperforming" in query or "best" in query:
                refined_query = "top performing stocks today"
            elif "stock" in query:
                refined_query = "stock market today"
            elif "moving average" in query or "prediction algorithms" in query:
                refined_query = "moving average financial markets prediction algorithms recent results"
            else:
                refined_query = "financial market news today"
        print(f"Refined query: {refined_query}")

        # Perform initial web search
        search_results = web_search(refined_query)
        print(f"Search results for {refined_query}: {search_results}")

        # Track thinking process
        thinking_steps = [
            f"Initial query: {request.query}",
            f"Refined query: {refined_query}",
            f"Initial search results: {search_results}"
        ]

        # Combine system prompt, history, query, and search results
        prompt = f"{FINANCE_PROMPT}\nConversation history: {request.history}\nUser query: {request.query}\nWeb search results: {search_results}"

        # Initial response with function calling
        response = chat.send_message(
            prompt,
            tools=[{"function_declarations": [web_search_declaration]}]
        )
        print(f"Initial response: {response}")

        iteration_count = 0
        max_iterations = 3
        while iteration_count < max_iterations:
            if hasattr(response.parts[0], "function_call") and response.parts[0].function_call:
                function_call = response.parts[0].function_call
                if function_call.name == "web_search":
                    query = function_call.args["query"]
                    print(f"Performing follow-up search for: {query}")
                    search_results = web_search(query)
                    print(f"Follow-up search results: {search_results}")
                    thinking_steps.append(f"Follow-up search for '{query}': {search_results}")
                    prompt = f"{FINANCE_PROMPT}\nConversation history: {request.history}\nUser query: {request.query}\nWeb search results: {search_results}"
                    response = chat.send_message(
                        prompt,
                        tools=[{"function_declarations": [web_search_declaration]}]
                    )
                    print(f"Follow-up response: {response}")
                    iteration_count += 1
                else:
                    raise HTTPException(status_code=400, detail="Unknown function call")
            else:
                final_response = response.text
                thinking_process = "\n".join(thinking_steps)
                print(f"Thinking process: {thinking_process}")
                print(f"Final response: {final_response}")
                # Update history with new query and response
                updated_history = request.history + [
                    Message(role="user", content=request.query),
                    Message(role="bot", content=final_response)
                ]
                return {"response": final_response, "thinking": thinking_process, "history": updated_history}
        raise HTTPException(status_code=500, detail="Max iterations reached without final response")
    except Exception as e:
        print(f"Error in deep_search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

def get_top_links(search: str, result: int = 10) -> list[str]:
    """
    Get the top links from Google search results using ScrapingDog API.
    """
    url = "https://api.scrapingdog.com/google"
    params = {
        "api_key": os.getenv("SCRAP_KEY"),
        "query": search,
        "results": result,
        "country": "us",
        "page": 0,
        "advance_search": "false"
    }
    try:
        response = requests.get(url, params=params, timeout=10) 
        response.raise_for_status()  
        data = response.json()

        links = []
        if "organic_results" in data:
            for res in data["organic_results"]:
                if "link" in res:
                    links.append(res["link"])
        return links
    except requests.exceptions.RequestException as e:
        print(f"Error fetching links from ScrapingDog: {e}")
        return []

@app.get('/links')
async def get_search_query_fastapi(
    search: str = Query(..., description="Search term for Google"),
    result: int = Query(10, description="Number of results to retrieve")
):
    """
    Endpoint to get the search query from the user and initiate crawling.
    """
    try:
        links = get_top_links(search, result)
        if not links:
            raise HTTPException(status_code=500, detail="Could not retrieve links from search.")
        
        await crawl_parallel(links, search)
        return JSONResponse(content={'message': 'Search query was entered successfully'}, status_code=200)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post('/query')
async def query_fastapi(request: Request):
    """
    Endpoint to handle the query from the user.
    """
    try:
        data_request = await request.json()
        query = data_request.get('query')
        if not query:
            raise HTTPException(status_code=400, detail='Query Not Provided')
        
        answer = cache.ask(query, rag_chain)
        return JSONResponse(content={'answer': answer})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post('/report')
async def report_fastapi(request: Request):
    """
    Endpoint to handle a cache-response report from the user.
    """
    try:
        data_request = await request.json()
        if not data_request or 'user_query' not in data_request:
            raise HTTPException(status_code=400, detail='user_query not provided in request body')

        question = data_request.get('user_query')
        cache.report_update(question)
        return JSONResponse(content={'message': 'User query reported successfully'}, status_code=200)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in /report: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4200)