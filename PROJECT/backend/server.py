import os
import warnings
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import requests
from dotenv import load_dotenv
# Your local imports
from app import rag_pipeline, SemanticCache, deep_search_pipeline
from crawler import crawl_parallel
# Import the new FastAPI router from deep_search.py
from deep_search import router as deep_search_router

# Environment & App setup
load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = FastAPI(title="Unified Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount deep_search router
app.include_router(deep_search_router, prefix="", tags=["deep_search"])


data_from_files = "crawled_data"
rag_chain = rag_pipeline(data_from_files)
cache = SemanticCache()


# /links endpoint (was Flask @app.route('/links', methods=['GET']))

@app.get("/links")
async def get_search_query(
    search: str = Query(None, description="Search term"),
    result: int = Query(5, description="Number of results"),
):
    """
    Get the top links from Google search via ScrapingDog API,
    then crawl them in parallel.
    """
    if not search:
        raise HTTPException(status_code=400, detail="Search term not provided")

    try:
        # Fetch top links
        url = "https://api.scrapingdog.com/google"
        params = {
            "api_key": os.getenv("SCRAP_KEY"),
            "query": search,
            "results": result,
            "country": "us",
            "page": 0,
            "advance_search": "false",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        links = [
            item["link"]
            for item in data.get("organic_results", [])
            if "link" in item
        ]

        # Kick off your crawler
        await crawl_parallel(links, search)

        return {"message": "Search query was entered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# /query endpoint (was Flask @app.route('/query', methods=['POST']))
@app.post("/query")
async def query_endpoint(request: Request):
    """
    Handle a simple RAG query using your SemanticCache.
    """
    try:
        payload = await request.json()
        query = payload.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query not provided")

        answer = cache.ask(query, rag_chain)
        return {"answer": answer}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# /report endpoint (was Flask @app.route('/report', methods=['POST']))
@app.post("/report")
async def report_endpoint(request: Request):
    """
    Handle a cache-report update from the user.
    """
    try:
        payload = await request.json()
        user_query = payload.get("user_query")
        if not user_query:
            raise HTTPException(status_code=400, detail="`user_query` not provided")

        cache.report_update(user_query)
        return {"message": "User query reported successfully"}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=4200,
        reload=True,   # remove in production
    )