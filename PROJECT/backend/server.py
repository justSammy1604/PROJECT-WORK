import uvicorn
import requests
import os
import warnings
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List 
from datetime import datetime
from app import rag_pipeline, SemanticCache, deep_search_pipeline # Assuming 'app' refers to a local package or module
from dotenv import load_dotenv
from crawler import crawl_parallel
# from deep_search import FINANCE_PROMPT, web_search, model, SearchRequest, Message, web_search_declaration
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

data_from_files = 'crawled_data'
rag_chain = rag_pipeline(data_from_files)
cache = SemanticCache()

@app.route('/deep_search', methods=['POST'])
def deep_search():
    """Endpoint for deep search functionality using SerpAPI and Gemini."""
    try:
        data_request = requests.json

        if not data_request or 'query' not in data_request:
            return JSONResponse(content={
                "success": False,
                "error": "Query parameter is required"
            }, status_code=400)

        query = data_request['query'].strip()
        if not query:
            return JSONResponse(content={
                "success": False,
                "error": "Query cannot be empty"
            }, status_code=400)

        # Optional parameters
        max_searches = data_request.get('max_searches', 5)
        confidence_threshold = data_request.get('confidence_threshold', 85)

        # Perform deep search using the function from app.py
        result = deep_search_pipeline(query, max_searches, confidence_threshold)

        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=500)

    except Exception as e:
        app.logger.error(f"Error in /deep-search: {str(e)}")
        return JSONResponse(content={
            "success": False,
            "error": f"Server error: {str(e)}"
        }, status_code=500)

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
async def query_fastapi(requests: Request):
    """
    Endpoint to handle the query from the user.
    """
    try:
        data_request = await requests.json()
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
async def report_fastapi(requests: Request):
    """
    Endpoint to handle a cache-response report from the user.
    """
    try:
        data_request = await requests.json()
        if not data_request or 'user_query' not in data_request:
            raise HTTPException(status_code=400, detail='user_query not provided in requests body')

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