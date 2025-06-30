import os
import json
import warnings
import time
import faiss
import requests
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from tools import agent_response
from pypdf import PdfReader
import sys
from unicodedata import category
import spacy
import re
import requests
from typing import List, Dict, Any


warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
hf_key = os.getenv('HF_TOKEN')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

# Deep Search Configuration
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
GEMINI_API_KEY = os.getenv('GOOGLE_API_MODEL')  # Using same key as your existing setup

# Semantic Cache Implementation
def init_cache():
    index = faiss.IndexFlatL2(768)
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", use_auth_token=hf_key)
    return index, encoder

def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache_data = json.load(file)
        cache = {
            "questions": OrderedDict(cache_data.get("questions", {})),
            "response_text": cache_data.get("response_text", []),
            "frequencies": cache_data.get("frequencies", {}),
            "reports": cache_data.get("reports", {})
        }
    except FileNotFoundError:
        cache = {"questions": OrderedDict(), "response_text": [], "frequencies": {}, "reports": {}}
    return cache

def store_cache(json_file, cache):
    print(f"Saving to {json_file}")
    with open(json_file, "w") as file:
        json.dump(cache, file)

# Deep Search Engine Class
class DeepSearchEngine:
    def __init__(self, max_searches=5):
        self.max_searches = max_searches
        self.search_results = []
        self.search_queries = []

    def search_web(self, query: str) -> Dict[str, Any]:
        """Perform a web search using SerpAPI"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "num": 10,
                "gl": "in",
                "hl": "en"
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"error": str(e)}

    def extract_relevant_info(self, search_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract relevant information from search results"""
        relevant_info = []

        # Extract organic results
        if "organic_results" in search_data:
            for result in search_data["organic_results"][:5]:  # Top 5 results
                info = {
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "source": result.get("source", "")
                }
                relevant_info.append(info)

        # Extract knowledge graph if available
        if "knowledge_graph" in search_data:
            kg = search_data["knowledge_graph"]
            info = {
                "title": kg.get("title", ""),
                "snippet": kg.get("description", ""),
                "link": "",
                "source": "Knowledge Graph"
            }
            relevant_info.append(info)

        # Extract answer box if available
        if "answer_box" in search_data:
            ab = search_data["answer_box"]
            info = {
                "title": ab.get("title", ""),
                "snippet": ab.get("snippet", "") or ab.get("answer", ""),
                "link": ab.get("link", ""),
                "source": "Answer Box"
            }
            relevant_info.append(info)

        return relevant_info

    def analyze_with_gemini(self, query: str, search_results: List[Dict], search_count: int, confidence_threshold: int = 85) -> Dict[str, Any]:
        """Use Gemini to analyze search results and determine next steps"""
        try:
            # Prepare context
            context = f"""
            Original Query: {query}
            Search Count: {search_count}/{self.max_searches}
            
            Search Results:
            """

            for i, result_set in enumerate(search_results):
                context += f"\n--- Search {i+1} Results ---\n"
                for result in result_set:
                    context += f"Title: {result['title']}\n"
                    context += f"Snippet: {result['snippet']}\n"
                    context += f"Source: {result['source']}\n\n"

            prompt = f"""
            {context}
            
            Based on the search results above, please analyze:
            
            1. Do we have enough information to provide a comprehensive answer to the original query?
            2. If not, what specific aspect should we search for next?
            3. Provide a confidence score (0-100) for how well the current information answers the query.
            
            Respond in JSON format:
            {{
                "sufficient_info": boolean,
                "confidence_score": number,
                "next_search_query": "string or null",
                "reasoning": "string explaining your analysis"
            }}
            """

            # Use your existing model instance
            response = model.invoke(prompt)

            # Parse JSON response
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "sufficient_info": search_count >= 3,
                    "confidence_score": 70,
                    "next_search_query": None,
                    "reasoning": "Analysis completed with available data"
                }

        except Exception as e:
            print(f"Gemini analysis error: {str(e)}")
            return {
                "sufficient_info": True,
                "confidence_score": 60,
                "next_search_query": None,
                "reasoning": f"Error in analysis: {str(e)}"
            }

    def generate_final_report(self, query: str, all_results: List[List[Dict]], search_queries: List[str]) -> str:
        """Generate comprehensive report using Gemini"""
        try:
            # Prepare comprehensive context
            context = f"Original Query: {query}\n\n"

            for i, (search_query, results) in enumerate(zip(search_queries, all_results)):
                context += f"=== Search {i+1}: '{search_query}' ===\n"
                for result in results:
                    context += f"• {result['title']}\n"
                    context += f"  {result['snippet']}\n"
                    context += f"  Source: {result['source']}\n\n"

            prompt = f"""
            {context}
            
            Based on all the search results above, please generate a comprehensive, well-structured report that answers the original query: "{query}"
            
            Requirements:
            1. Provide a clear, detailed answer focused on financial markets and investment insights
            2. Include relevant facts and data points
            3. Cite sources where appropriate
            4. Structure the information logically with sections and subsections
            5. Highlight key insights and actionable information
            6. If there are conflicting information, mention it
            7. Make it informative and easy to read for financial analysis
            8. Focus on US financial markets when applicable
            
            Format the response as a structured report with clear headings and bullet points where appropriate.
            """

            response = model.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Error generating report: {str(e)}"

    def deep_search(self, query: str, confidence_threshold: int = 85) -> Dict[str, Any]:
        """Perform deep search with iterative refinement"""
        self.search_results = []
        self.search_queries = []

        current_query = query
        search_count = 0

        try:
            while search_count < self.max_searches:
                search_count += 1
                print(f"Performing search {search_count}: {current_query}")

                # Perform search
                search_data = self.search_web(current_query)
                if "error" in search_data:
                    break

                # Extract relevant information
                relevant_info = self.extract_relevant_info(search_data)
                self.search_results.append(relevant_info)
                self.search_queries.append(current_query)

                # Analyze with Gemini (except on last iteration)
                if search_count < self.max_searches:
                    analysis = self.analyze_with_gemini(query, self.search_results, search_count, confidence_threshold)

                    if analysis["sufficient_info"] or analysis["confidence_score"] >= confidence_threshold:
                        print(f"Sufficient information found after {search_count} searches")
                        break

                    if analysis["next_search_query"]:
                        current_query = analysis["next_search_query"]
                    else:
                        break

                # Small delay to avoid rate limiting
                time.sleep(1)

            # Generate final report
            final_report = self.generate_final_report(query, self.search_results, self.search_queries)

            return {
                "success": True,
                "query": query,
                "searches_performed": search_count,
                "search_queries": self.search_queries,
                "report": final_report,
                "raw_results": self.search_results
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "searches_performed": search_count
            }

# Deep Search Pipeline Function
def deep_search_pipeline(query: str, max_searches: int = 5, confidence_threshold: int = 85) -> Dict[str, Any]:
    """
    Main function to perform deep search - called by the Flask endpoint
    """
    try:
        # Check for required environment variables
        if not SERPAPI_KEY:
            return {
                "success": False,
                "error": "SERPAPI_KEY environment variable not set"
            }

        # Initialize search engine
        search_engine = DeepSearchEngine(max_searches=max_searches)

        # Perform deep search
        result = search_engine.deep_search(query, confidence_threshold)

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Deep search pipeline error: {str(e)}"
        }

class SemanticCache:
    def __init__(self, json_file="cache_file.json", threshold=0.35, max_response=100): #max_response is useless but I didn't remove it
        self.index, self.encoder = init_cache()
        self.threshold = threshold
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.question_addition_index = {q: i for i, q in enumerate(self.cache["questions"].keys())}
        if self.cache["questions"]:
            embeddings = self.encoder.encode(list(self.cache["questions"].keys()))
            self.index.add(embeddings)

    # def evict(self):
    #     while len(self.cache["questions"]) > self.max_response:
    #         self.cache["questions"].popitem(last=False)
    #         self.cache["response_text"].pop(0)

    def report_update(self, question: str) ->str:
        embedding = self.encoder.encode([question])

        D, I = self.index.search(embedding, 1)
        if len(I[0]) > 0 and I[0][0] >= 0 and D[0][0] <= self.threshold:
            row_id = int(I[0][0])
            matched_question = list(self.cache["questions"].keys())[row_id]
            self.cache["reports"][matched_question] = self.cache["reports"].get(matched_question, 0) + 1
            print("report updated")

    def ask(self, question: str, rag_chain) -> str:
        start_time = time.time()
        embedding = self.encoder.encode([question])

        D, I = self.index.search(embedding, 1)
        if len(I[0]) > 0 and I[0][0] >= 0 and D[0][0] <= self.threshold:
            row_id = int(I[0][0])
            response_text = self.cache["response_text"][row_id]
            matched_question = list(self.cache["questions"].keys())[row_id]
            self.cache["frequencies"][matched_question] = self.cache["frequencies"].get(matched_question, 0) + 1
            self.cache["questions"].move_to_end(matched_question)
            print("Response came from CACHE.")
        else:
            response_text = query_response(question, rag_chain)  # Use the existing query_response function
            self.cache["frequencies"][question] = 1
            self.cache["reports"][question] = 0
            self.question_addition_index[question] = len(self.cache["response_text"])
            self.cache["questions"][question] = None
            self.cache["response_text"].append(response_text)
            self.index.add(embedding)
            print("Response came from VECTORSTORE/RAG.")

        # self.evict()
        store_cache(self.json_file, self.cache)
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        return response_text

    def cleanup_cache(self):
        low_freq_questions = [(q, self.cache["frequencies"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["frequencies"].get(q, 0) < 2]
        high_freq_questions = [(q, self.cache["frequencies"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["frequencies"].get(q, 0) >= 2]
        low_reptd_questions = [(q, self.cache["reports"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["reports"].get(q, 0) < 100]
        low_freq_questions.sort(key=lambda x: list(self.cache["questions"].keys()).index(x[0]))
        kept_questions = [q for q, freq in high_freq_questions]
        kept_questions = [q for q in kept_questions and [q for q, reptd in low_reptd_questions]]#questionable

        if not kept_questions:
            self.cache["questions"] = OrderedDict()
            self.cache["response_text"] = []
            self.cache["frequencies"] = {}
            self.cache["reports"] = {}
        else:
            new_questions = OrderedDict()
            new_response_text = []
            new_frequencies = {}
            new_reports = {}
            for q in kept_questions:
                new_questions[q] = None
                new_response_text.append(self.cache["response_text"][self.question_addition_index[q]])
                new_frequencies[q] = self.cache["frequencies"][q]
                new_reports[q] = self.cache["reports"][q]
            self.cache["questions"] = new_questions
            self.cache["response_text"] = new_response_text
            self.cache["frequencies"] = new_frequencies
            self.cache["reports"] = new_reports

        store_cache(self.json_file, self.cache)
        print("Cache cleaned up. Low-frequency questions (< 2) removed, frequencies preserved in cache_file.json.")

# Deep Search Engine Class
class DeepSearchEngine:
    def __init__(self, max_searches=5):
        self.max_searches = max_searches
        self.search_results = []
        self.search_queries = []
        
    def search_web(self, query: str) -> Dict[str, Any]:
        """Perform a web search using SerpAPI"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "num": 10,
                "gl": "in",
                "hl": "en"
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"error": str(e)}
    
    def extract_relevant_info(self, search_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract relevant information from search results"""
        relevant_info = []
        
        # Extract organic results
        if "organic_results" in search_data:
            for result in search_data["organic_results"][:5]:  # Top 5 results
                info = {
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "source": result.get("source", "")
                }
                relevant_info.append(info)
        
        # Extract knowledge graph if available
        if "knowledge_graph" in search_data:
            kg = search_data["knowledge_graph"]
            info = {
                "title": kg.get("title", ""),
                "snippet": kg.get("description", ""),
                "link": "",
                "source": "Knowledge Graph"
            }
            relevant_info.append(info)
        
        # Extract answer box if available
        if "answer_box" in search_data:
            ab = search_data["answer_box"]
            info = {
                "title": ab.get("title", ""),
                "snippet": ab.get("snippet", "") or ab.get("answer", ""),
                "link": ab.get("link", ""),
                "source": "Answer Box"
            }
            relevant_info.append(info)
            
        return relevant_info
    
    def analyze_with_gemini(self, query: str, search_results: List[Dict], search_count: int, confidence_threshold: int = 85) -> Dict[str, Any]:
        """Use Gemini to analyze search results and determine next steps"""
        try:
            # Prepare context
            context = f"""
            Original Query: {query}
            Search Count: {search_count}/{self.max_searches}
            
            Search Results:
            """
            
            for i, result_set in enumerate(search_results):
                context += f"\n--- Search {i+1} Results ---\n"
                for result in result_set:
                    context += f"Title: {result['title']}\n"
                    context += f"Snippet: {result['snippet']}\n"
                    context += f"Source: {result['source']}\n\n"
            
            prompt = f"""
            {context}
            
            Based on the search results above, please analyze:
            
            1. Do we have enough information to provide a comprehensive answer to the original query?
            2. If not, what specific aspect should we search for next?
            3. Provide a confidence score (0-100) for how well the current information answers the query.
            
            Respond in JSON format:
            {{
                "sufficient_info": boolean,
                "confidence_score": number,
                "next_search_query": "string or null",
                "reasoning": "string explaining your analysis"
            }}
            """
            
            # Use your existing model instance
            response = model.invoke(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "sufficient_info": search_count >= 3,
                    "confidence_score": 70,
                    "next_search_query": None,
                    "reasoning": "Analysis completed with available data"
                }
                
        except Exception as e:
            print(f"Gemini analysis error: {str(e)}")
            return {
                "sufficient_info": True,
                "confidence_score": 60,
                "next_search_query": None,
                "reasoning": f"Error in analysis: {str(e)}"
            }
    
    def generate_final_report(self, query: str, all_results: List[List[Dict]], search_queries: List[str]) -> str:
        """Generate comprehensive report using Gemini"""
        try:
            # Prepare comprehensive context
            context = f"Original Query: {query}\n\n"
            
            for i, (search_query, results) in enumerate(zip(search_queries, all_results)):
                context += f"=== Search {i+1}: '{search_query}' ===\n"
                for result in results:
                    context += f"• {result['title']}\n"
                    context += f"  {result['snippet']}\n"
                    context += f"  Source: {result['source']}\n\n"
            
            prompt = f"""
            {context}
            
            Based on all the search results above, please generate a comprehensive, well-structured report that answers the original query: "{query}"
            
            Requirements:
            1. Provide a clear, detailed answer focused on financial markets and investment insights
            2. Include relevant facts and data points
            3. Cite sources where appropriate
            4. Structure the information logically with sections and subsections
            5. Highlight key insights and actionable information
            6. If there are conflicting information, mention it
            7. Make it informative and easy to read for financial analysis
            8. Focus on US financial markets when applicable
            
            Format the response as a structured report with clear headings and bullet points where appropriate.
            """
            
            response = model.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def deep_search(self, query: str, confidence_threshold: int = 85) -> Dict[str, Any]:
        """Perform deep search with iterative refinement"""
        self.search_results = []
        self.search_queries = []
        
        current_query = query
        search_count = 0
        
        try:
            while search_count < self.max_searches:
                search_count += 1
                print(f"Performing search {search_count}: {current_query}")
                
                # Perform search
                search_data = self.search_web(current_query)
                if "error" in search_data:
                    break
                
                # Extract relevant information
                relevant_info = self.extract_relevant_info(search_data)
                self.search_results.append(relevant_info)
                self.search_queries.append(current_query)
                
                # Analyze with Gemini (except on last iteration)
                if search_count < self.max_searches:
                    analysis = self.analyze_with_gemini(query, self.search_results, search_count, confidence_threshold)
                    
                    if analysis["sufficient_info"] or analysis["confidence_score"] >= confidence_threshold:
                        print(f"Sufficient information found after {search_count} searches")
                        break
                    
                    if analysis["next_search_query"]:
                        current_query = analysis["next_search_query"]
                    else:
                        break
                
                # Small delay to avoid rate limiting
                time.sleep(1)
            
            # Generate final report
            final_report = self.generate_final_report(query, self.search_results, self.search_queries)
            
            return {
                "success": True,
                "query": query,
                "searches_performed": search_count,
                "search_queries": self.search_queries,
                "report": final_report,
                "raw_results": self.search_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "searches_performed": search_count
            }

# Deep Search Pipeline Function
def deep_search_pipeline(query: str, max_searches: int = 5, confidence_threshold: int = 85) -> Dict[str, Any]:
    """
    Main function to perform deep search - called by the Flask endpoint
    """
    try:
        # Check for required environment variables
        if not SERPAPI_KEY:
            return {
                "success": False,
                "error": "SERPAPI_KEY environment variable not set"
            }
        
        # Initialize search engine
        search_engine = DeepSearchEngine(max_searches=max_searches)
        
        # Perform deep search
        result = search_engine.deep_search(query, confidence_threshold)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Deep search pipeline error: {str(e)}"
        }

# Original LangChain Functions (unchanged)
def split_text(documents):
    text_split = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_split.split_documents(documents)
    return docs

def vectordb_information(docs):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    return vectorstore

def rag_model(vectorstore):
    # 1. Standard Condense Question Prompt (for history handling)
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # 2. Define QA Prompts (accepting context and question)
    template_data_driven_qa = """
    You are a professional financial advisor assistant specializing in the United States financial markets, including stocks, cryptocurrency, bonds, and economic news.
    Tone: Use a formal and professional tone in your responses.
    Sources: Use only the provided context and any relevant web-retrieved financial data. Do not hallucinate or invent information not present in these sources.
    Provided Context:
    {context}
    Task: Answer the user's question by summarizing, analyzing, or explaining based on the available financial data (including real-time market data), news, and the provided context.
    If not available: If the needed information is not found in the provided sources, respond formally indicating that it is not available.

    User Question: {question}
    Answer: """
    PROMPT_DATA_DRIVEN = PromptTemplate(template=template_data_driven_qa, input_variables=["context", "question"])

    template_strict_qa = """
    You are a professional financial advisor assistant focused on providing accurate and insightful answers based solely on the provided document context.
    Tone: Maintain a formal, respectful, and professional tone at all times.
    Source: Only utilize information contained within the provided documents (context). Do not fabricate or infer information that is not explicitly stated.
    Provided Context:
    {context}
    Task: Based on the user's question, offer well-structured, data-backed advice or summaries related to financial markets in the United States, including but not limited to stock markets, cryptocurrencies, bonds, personal finance, investment strategies, and economic policies, using *only* the provided context.
    Limitation: If the documents do not contain sufficient information to answer the user's query, politely inform the user that the requested information is unavailable based on the current data.
    User Question: {question}
    Answer:"""
    PROMPT_STRICT = PromptTemplate(template=template_strict_qa, input_variables=["context", "question"])

    # 3. Create two separate QA chains (combine_docs_chain)
    qa_chain_data_driven_instance = load_qa_chain(
        model, # Use your LLM model here
        chain_type="stuff", # Or "map_reduce", "refine", etc.
        prompt=PROMPT_DATA_DRIVEN
    )

    qa_chain_strict_instance = load_qa_chain(
        model, # Use your LLM model here
        chain_type="stuff",
        prompt=PROMPT_STRICT
    )

    # Create PromptTemplate objects
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question',
        output_key='answer',
        return_messages=True
    )

    # 5. Create the ConversationalRetrievalChains, passing the correct components
    conv_chain_data_driven = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        question_generator=LLMChain(llm=model, prompt=CONDENSE_QUESTION_PROMPT), # Handles history
        combine_docs_chain=qa_chain_data_driven_instance, # Handles QA with context
        memory=memory, # Use shared or specific memory
        return_source_documents=True,
        verbose=True
    )

    conv_chain_strict = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        question_generator=LLMChain(llm=model, prompt=CONDENSE_QUESTION_PROMPT), # Handles history
        combine_docs_chain=qa_chain_strict_instance, # Handles QA with context
        memory=memory, # Use shared or specific memory
        return_source_documents=True,
        verbose=True
    )

    return {"data_driven": conv_chain_data_driven, "strict": conv_chain_strict}

# Modify query_response slightly for clarity (though original logic was okay)
def query_response(query, rag_chain_dict): # Renamed rag_chain to rag_chain_dict
    try:
        # Check for |||TRUE||| in the query
        enable_data_driven = bool(re.search(r"\|\|\|TRUE\|\|\|", query, re.IGNORECASE))

        # Remove |||TRUE||| from the query
        cleaned_query = re.sub(r"\|\|\|true\|\|\|", "", query, flags=re.IGNORECASE).strip()

        # Select the appropriate chain based on the flag
        if enable_data_driven:
            print("Using Data Driven Chain")
            # selected_chain = rag_chain_dict["data_driven"]
            return agent_response(cleaned_query)  # Use the agent_response function directly
        else:
            print("Using Strict Chain")
            selected_chain = rag_chain_dict["strict"]

        result = selected_chain({"question": cleaned_query})
        response = result['answer']
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def load_and_process(doc_source):
    all_docs = []
    loader = DirectoryLoader(doc_source, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()
    all_docs.extend(documents)
    return split_text(sw_rem_main(all_docs))

def rag_pipeline(document_sources):
    processed_docs = load_and_process(document_sources)
    vector_store = vectordb_information(processed_docs)
    rag_chain = rag_model(vector_store)
    return rag_chain

def conver(temp):
    temp2 = ''
    for text in temp:
        temp2 += text + ' '
    return temp2

def sw_rem_main(all_docs):
    temp = []
    for doc in all_docs:
        temp.append(doc.page_content)

    swrem = stopword_removal(temp)

    for doc in all_docs:
        doc.page_content = conver(swrem.pop(0))
    return all_docs

def stopword_removal(texts):
    documents = []
    nlp = spacy.load("en_core_web_sm")
    codepoints = range(sys.maxunicode + 1)
    punctuation = {c for i in codepoints if category(c := chr(i)).startswith("P")} | {"\n", " "}

    for text in texts:
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        for i in punctuation:
            for token in filtered_tokens.copy():
                if i in token:
                    filtered_tokens.remove(token)
        documents.append(filtered_tokens)
    return documents
