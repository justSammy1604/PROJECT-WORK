from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Verify environment variables
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in .env")
if not os.getenv("SERPAPI_API_KEY"):
    raise ValueError("SERPAPI_API_KEY not found in .env")

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {str(e)}")
    raise

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class SearchRequest(BaseModel):
    query: str

# Web search function
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

# Function declaration for Gemini
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

# Updated Finance-specific system prompt
FINANCE_PROMPT = """
You are a financial research assistant specializing in stock markets, cryptocurrencies, and economic trends. When a user asks about 'the market,' 'stocks,' or similar terms, assume they mean financial markets unless specified otherwise. Your task is to provide a detailed, narrative response using the provided web search results. Follow these steps:
[Todays Date 05-05-2025 ]
1. **Analyze the Query**: Understand the user's intent. If the query involves technical concepts like moving averages or prediction algorithms, provide a thorough explanation of the concepts, their applications in financial markets, and comparisons with other methods.
2. **Extract Relevant Information**: From the search results, pull out key details such as:
   - Definitions and explanations of financial concepts (e.g., moving averages, prediction algorithms).
   - Examples of how these concepts are applied in trading or forecasting.
   - Comparisons between different methods or tools (e.g., moving averages vs. machine learning models).
   - Any limitations or challenges associated with these methods.
3. **Expand with Context**: If the search results are limited, use general knowledge to provide a comprehensive overview. Include:
   - Practical examples of how the concepts are used in financial markets.
   - Comparisons between traditional methods (e.g., moving averages) and modern approaches (e.g., machine learning).
   - Insights into their effectiveness, reliability, and limitations.
4. **Structure the Response**: Write a detailed, narrative response in markdown format (using headings, bullet points, and bold text for clarity). Do NOT return a JSON response. The response should be at least 500 words, resembling a DeepSearch reply, with the following sections:
   - **Introduction**: Briefly introduce the topic and its relevance to financial markets.
   - **Detailed Explanation**: Explain the main concept (e.g., moving averages) in depth, including how it works and its applications.
   - **Comparison with Other Methods**: Compare the main concept with other prediction algorithms (e.g., machine learning models like LSTM).
   - **Limitations and Challenges**: Discuss any drawbacks or limitations of the methods.
   - **Conclusion**: Summarize the key points and provide a final thought or recommendation.
5. **Handle Missing Data**: If specific data (e.g., current market performance) is unavailable, note it explicitly and focus on explaining the concepts and their applications.

Ensure the response is engaging, informative, and suitable for someone looking to understand financial prediction methods.
"""

@app.post("/deepsearch")
async def deep_search(request: SearchRequest):
    try:
        print(f"Received query: {request.query}")
        
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

        # Combine system prompt, query, and search results
        prompt = f"{FINANCE_PROMPT}\nUser query: {request.query}\nWeb search results: {search_results}"
        
        response = model.generate_content(
            prompt,
            tools=[{"function_declarations": [web_search_declaration]}]
        )
        print(f"Initial response: {response.candidates}")

        iteration_count = 0
        max_iterations = 3
        while iteration_count < max_iterations:
            if hasattr(response.candidates[0].content.parts[0], "function_call") and response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                if function_call.name == "web_search":
                    query = function_call.args["query"]
                    print(f"Performing follow-up search for: {query}")
                    search_results = web_search(query)
                    print(f"Follow-up search results: {search_results}")
                    thinking_steps.append(f"Follow-up search for '{query}': {search_results}")
                    prompt = f"{FINANCE_PROMPT}\nUser query: {request.query}\nWeb search results: {search_results}"
                    response = model.generate_content(
                        prompt,
                        tools=[{"function_declarations": [web_search_declaration]}]
                    )
                    print(f"Follow-up response: {response.candidates}")
                    iteration_count += 1
                else:
                    raise HTTPException(status_code=400, detail="Unknown function call")
            else:
                final_response = response.candidates[0].content.parts[0].text
                thinking_process = "\n".join(thinking_steps)
                print(f"Thinking process: {thinking_process}")
                print(f"Final response: {final_response}")
                return {"response": final_response, "thinking": thinking_process}
        raise HTTPException(status_code=500, detail="Max iterations reached without final response")
    except Exception as e:
        print(f"Error in deep_search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")