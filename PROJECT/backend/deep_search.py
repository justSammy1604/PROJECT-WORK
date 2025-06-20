# Flask Version of DeepSearch Gemini Integration
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from typing import List
from datetime import datetime
import os
import google.generativeai as genai
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in .env")
if not os.getenv("SERPAPI_API_KEY"):
    raise ValueError("SERPAPI_API_KEY not found in .env")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class SearchRequest(BaseModel):
    query: str
    history: List[Message] = []

def web_search(query: str) -> str:
    try:
        search = GoogleSearch({"q": query, "api_key": os.getenv("SERPAPI_API_KEY")})
        result = search.get_dict()
        organic_results = result.get('organic_results', [])
        snippets = [item['snippet'] for item in organic_results[:5]]
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        print(f"SerpAPI error: {str(e)}")
        return f"Search failed: {str(e)}"

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

model = genai.GenerativeModel("gemini-2.0-flash")
todays_date = datetime.today().strftime("%d-%m-%Y")

FINANCE_PROMPT = f"""
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

@app.route("/deepsearch", methods=["POST"])
def deep_search():
    try:
        body = request.get_json()
        search_request = SearchRequest(**body)

        print(f"Received query: {search_request.query}")

        history = [
            {"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]}
            for msg in search_request.history
        ]

        chat = model.start_chat(history=history)
        query = search_request.query.lower()
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

        search_results = web_search(refined_query)
        thinking_steps = [
            f"Initial query: {search_request.query}",
            f"Refined query: {refined_query}",
            f"Initial search results: {search_results}"
        ]

        prompt = f"{FINANCE_PROMPT}\nConversation history: {search_request.history}\nUser query: {search_request.query}\nWeb search results: {search_results}"

        response = chat.send_message(
            prompt,
            tools=[{"function_declarations": [web_search_declaration]}]
        )

        iteration_count = 0
        max_iterations = 3

        while iteration_count < max_iterations:
            if hasattr(response.parts[0], "function_call") and response.parts[0].function_call:
                function_call = response.parts[0].function_call
                if function_call.name == "web_search":
                    query = function_call.args["query"]
                    search_results = web_search(query)
                    thinking_steps.append(f"Follow-up search for '{query}': {search_results}")
                    prompt = f"{FINANCE_PROMPT}\nConversation history: {search_request.history}\nUser query: {search_request.query}\nWeb search results: {search_results}"
                    response = chat.send_message(
                        prompt,
                        tools=[{"function_declarations": [web_search_declaration]}]
                    )
                    iteration_count += 1
                else:
                    return jsonify({"error": "Unknown function call"}), 400
            else:
                final_response = response.text
                thinking_process = "\n".join(thinking_steps)
                updated_history = search_request.history + [
                    Message(role="user", content=search_request.query),
                    Message(role="bot", content=final_response)
                ]
                return jsonify({"response": final_response, "thinking": thinking_process, "history": [msg.dict() for msg in updated_history]})

        return jsonify({"error": "Max iterations reached without final response"}), 500

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 422
    except Exception as e:
        print(f"Error in deep_search: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
