import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import requests
from app import rag_pipeline, SemanticCache
from dotenv import load_dotenv
from crawler import crawl_parallel
load_dotenv()
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


data_from_files = 'crawled_data'  # We can also add crawled_data file as input here. 
rag_chain = rag_pipeline(data_from_files)
cache = SemanticCache()

def get_top_links(search,result=50):
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
        if not data_request or 'response_text' not in data_request:
            return jsonify({'error': 'response_text not provided in request body'}), 400

        reported_text = data_request.get('response_text')
        cache.report_update(reported_text) 
        return jsonify({'message': 'Response reported successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error in /report: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200, debug=True)
