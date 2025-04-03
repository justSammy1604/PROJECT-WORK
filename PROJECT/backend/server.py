from flask import Flask, request, jsonify
from flask_cors import CORS 
import requests
from app import rag_pipeline, SemanticCache
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

data = 'crawled_data'  # We can also add crawled_data file as input here. 
rag_chain = rag_pipeline(data)
cache = SemanticCache()
@app.route('/links', methods=['GET'])
def get_top_links(search):
    url = "https://api.scrapingdog.com/google"
    params = {
    "api_key": os.getenv("SCRAP_KEY"),
    "query": search,
    "results": 20,
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

    if "menu_items" in data:
        for item in data["menu_items"]:
            if "link" in item:
                links.append(item["link"])

    if "peopleAlsoAskedFor" in data:
        for question in data["peopleAlsoAskedFor"]:
          if "link" in question:
            links.append(question["link"])

    if "relatedSearches" in data:
        for related in data['relatedSearches']:
            if "link" in related:
                links.append(related["link"])

    return links

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Query Not Provided'}), 400
        answer = cache.ask(query, rag_chain)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200, debug=True)
