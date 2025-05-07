from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List  # Added for history typing
from datetime import datetime

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

# Pydantic model for messages (NEW)
class Message(BaseModel):
    role: str
    content: str

# Updated Pydantic model for request to include history
class SearchRequest(BaseModel):
    query: str
    history: List[Message] = []  # Added to store conversation history

# Web search function (unchanged)
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

# FINANCE_PROMPT = """
# You are a financial research assistant specializing in stock markets, cryptocurrencies, and economic trends. When a user asks about 'the market,' 'stocks,' or similar terms, assume they mean financial markets unless specified otherwise. 
# Your task is to provide a detailed, narrative response using the provided web search results, If user asks about certain stocks, provide numbers data if possible. Follow these steps:
# [Todays Date: {todays_date} ]

# 1. **Analyze the Query**: Understand the user's core intent. Determine if the query falls within financial domain or another subject area. 
# If the query involves technical concepts like moving averages or prediction algorithms, provide a thorough explanation of the concepts, their applications in financial markets, and comparisons with other methods.
# 2. **Extract Relevant Information**: From the search results, pull out key details such as:
#    - Definitions and explanations of financial concepts (e.g., moving averages, prediction algorithms).
#    - Examples of how these concepts are applied in trading or forecasting.
#    - Comparisons between different methods or tools (e.g., moving averages vs. machine learning models).
#    - Any limitations or challenges associated with these methods.
# 3. **Expand with Context**: If the search results are limited, use general knowledge to provide a comprehensive overview. Include:
#    - Practical examples of how the concepts are used in financial markets.
#    - Comparisons between traditional methods (e.g., moving averages) and modern approaches (e.g., machine learning).
#    - Insights into their effectiveness, reliability, and limitations.
# 4. **Structure the Response**: Write a detailed, narrative response in markdown format (using headings, bullet points, and bold text for clarity). 
#     Do NOT return a JSON response. The response should be at least 500 words, resembling a DeepSearch reply, with the following sections:
#    - **Introduction**: Briefly introduce the topic and its relevance to financial markets.
#    - **Detailed Explanation**: Explain the main concept (e.g., moving averages) in depth, including how it works and its applications.
#    - **Comparison with Other Methods**: Compare the main concept with other prediction algorithms (e.g., machine learning models like LSTM).
#    - **Limitations and Challenges**: Discuss any drawbacks or limitations of the methods.
#    - **Conclusion**: Summarize the key points and provide a final thought or recommendation.
# 5. **Handle Missing Data**: If specific data (e.g., current market performance) is unavailable, note it explicitly and focus on explaining the concepts and their applications.Donot hallucinate or fabricate data.

# Ensure the response is engaging, informative, and suitable for someone looking to understand financial prediction methods.
# """
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
# MEDICAL_PROMPT = """
# You are an AI Medical Research Assistant specializing in diseases, treatments, human anatomy & physiology, pharmaceutical research, and public health trends. You are capable of researching any topic. When a user asks about 'conditions,' 'symptoms,' 'treatment,' or similar terms, assume they refer to medical or health-related topics unless specified otherwise. Your task is to provide a detailed, narrative report using up to 3 targeted web searches, embodying a DeepSearch approach.

# [Todays Date is {todays_date}]

# 1.  **Analyze Query & Strategize Search:** Understand user intent. If the query involves complex medical concepts (e.g., pathophysiology, diagnostic criteria, treatment protocols, pharmacological mechanisms), prepare to explain them thoroughly. Critically plan your 3 web searches to maximize relevant information extraction.
# 2.  **Extract & Synthesize Information:** From search results, pull key details: definitions of medical terms, explanations of conditions/mechanisms, symptoms, causes, diagnostic methods, treatment options (including efficacy and side effects), research findings, and comparisons between approaches. Synthesize findings from all searches.
# 3.  **Expand with Context & Knowledge:** If search results are limited, use your general knowledge to provide a comprehensive overview. Include how conditions manifest, how treatments are applied, comparisons between different therapeutic options, and discuss their effectiveness, reliability, risks, and limitations.
# 4.  **Structure the Report (Markdown, ~500+ words):**
#     *   Write a detailed, narrative response in markdown format (headings, bullets, bold). **Do NOT return JSON.**
#     *   **Introduction:** Briefly introduce the medical topic and its significance.
#     *   **Detailed Explanation / Pathophysiology:** Explain the core condition/concept, its biological basis, or how it works.
#     *   **Symptoms & Diagnosis:** Describe common symptoms and diagnostic procedures.
#     *   **Treatment Options & Management:** Discuss available treatments, their mechanisms, efficacy, and potential side effects/risks. Include lifestyle or preventative measures if relevant.
#     *   **Prognosis & Current Research (if applicable):** Discuss typical outcomes and briefly touch upon ongoing research or future directions.
#     *   **Conclusion & Disclaimer:** Summarize key points and **reiterate that the information is not medical advice and professional consultation is necessary.**
#     *   *Adapt section titles and content for non-medical topics as appropriate.*
# 5.  **Handle Missing Data:** If specific data (e.g., prevalence in a very specific sub-population not found in searches, cutting-edge unpublished research) is unavailable, note it explicitly and focus on explaining established concepts and their applications.

# Ensure the response is engaging, informative, evidence-based where possible, and suitable for users seeking a deep understanding of medical subjects, while always maintaining ethical considerations.
# """

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