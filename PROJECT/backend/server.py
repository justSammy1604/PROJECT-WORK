import os
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS 
import requests
from app import rag_pipeline, SemanticCache
from dotenv import load_dotenv
from crawler import crawl_parallel
from deep_search import web_search, web_search_declaration,FINANCE_PROMPT,SearchRequest, Message, model, ValidationError
load_dotenv()
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes
warnings.filterwarnings("ignore", category=DeprecationWarning)


data_from_files = 'crawled_data'  # We can also add crawled_data file as input here. 
rag_chain = rag_pipeline(data_from_files)
cache = SemanticCache()


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

def get_top_links(search,result=5):
    """ Get the top links from Google search results using ScrapingDog API.
    Function to get the top links from Google search results using ScrapingDog API."""
    url = "https://api.scrapingdog.com/google"
    params = {
    "api_key": os.getenv("SCRAP_KEY"),
    "query": search,
    "results": result,
    "country": "us",
    "page": 0,
    "advance_search": "false"
}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
    
    links = []
    if "organic_results" in data:
        for result in data["organic_results"]:
            if "link" in result:
                links.append(result["link"])
    return links
     
@app.route('/links', methods=['GET'])
async def get_search_query():
    """Endpoint to get the search query from the user."""
    try:
        search = request.args.get('search')
        result = request.args.get('result')  # Default to 50 results if not provided
        if not search and not result:
            return jsonify({'error': 'Search Term Not Provided'}), 400
        links = get_top_links(search,result)
        await crawl_parallel(links,search)
        return jsonify({'message':'Search Query was entered successfully'}) , 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Endpoint to handle the query from the user."""
    try:
        data_request = request.json
        query = data_request.get('query')
        if not query:
            return jsonify({'error': 'Query Not Provided'}), 400
        answer = cache.ask(query, rag_chain)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/report', methods=['POST'])
def report():
    """Endpoint to handle a cache-response report from the user."""
    try:
        data_request = request.json
        if not data_request or 'user_query' not in data_request:
            return jsonify({'error': 'user_query not provided in request body'}), 400

        question = data_request.get('user_query')
        cache.report_update(question)        
        return jsonify({'message': 'User query reported successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error in /report: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200, debug=True)
