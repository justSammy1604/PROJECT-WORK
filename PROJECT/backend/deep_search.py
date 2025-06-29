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

FINANCE_PROMPT = """
You are an AI Research Assistant specializing in financial markets (stocks, cryptocurrencies, economic trends), but capable of researching any topic. 
Assume queries about 'the market,' 'stocks,' etc., refer to financial markets unless specified. 
Your task is to provide a detailed, narrative report using max up to 3 targeted web searches, embodying a DeepSearch approach.

[Todays Date is {todays_date}]
give results based on the todays dateby default unless specified otherwise.
1.  **Analyze Query & Strategize Search:** Understand user intent. If technical concepts (e.g., moving averages, algorithms) are involved, prepare to explain them thoroughly. 
    Critically plan your 3 web searches to maximize relevant information extraction for the report.
2.  **Extract & Synthesize Information:** From search results, pull key details: definitions, explanations, examples of applications (especially in finance), comparisons between methods/tools, and any limitations. Synthesize findings from all searches.
3.  **Expand with Context & Knowledge:** If search results are limited, use your general knowledge to provide a comprehensive overview. 
    Include practical examples, compare traditional vs. modern approaches (e.g., moving averages vs. machine learning), and discuss effectiveness, reliability, and limitations.
4.  **Structure the Report (Markdown, ~100+ words) unless explicitly stated otherwise:**
    *   Write a detailed, narrative response in markdown format (headings, bullets, bold). **Do NOT return JSON.**
    *   **Leave *TWO* lines after each paragraph**.
    *   **show comparisons using a table:** If the user asks you for differences or comparisons then use a table to show your response.
    *   **Introduction:** Briefly introduce the topic and its relevance.
    *   **Detailed Explanation:** Explain the core concept(s) in depth, including mechanics and applications.
    *   **Comparative Analysis / Further Applications:** (As applicable) Compare with other methods or provide more examples.
    *   **Limitations and Challenges:** Discuss drawbacks.
    *   **Conclusion:** Summarize key points.
    *   *Adapt section titles and content for non-financial topics as appropriate.*
5.  **Handle Missing Data:** If specific data is unavailable from searches, note it explicitly and focus on explaining concepts and their applications.
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

@app.post("/deep_search")
async def deep_search(request: SearchRequest):
    try:
        print(f"=== STEP 1: Query Reception ===")
        print(f"Received query: {request.query}")
        print(f"History length: {len(request.history)}")
        
        # Convert history to Gemini-compatible format
        history = [
            {"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]}
            for msg in request.history
        ]

        # Start a chat session with history
        chat = model.start_chat(history=history)

        # Track thinking process and sources
        thinking_steps = [
            f"Step 1 - User query received: {request.query}",
            f"Step 2 - Starting Gemini-driven search strategy"
        ]
        all_sources = []  # Track all sources from searches

        print(f"=== STEP 2: Send Query to Gemini for Analysis ===")
        # Let Gemini analyze the query and decide on search strategy
        initial_prompt = f"""{FINANCE_PROMPT.format(todays_date=todays_date)}

User Query: {request.query}

You are tasked with providing a comprehensive analysis of this query. Since you have access to the web_search function, please:

1. Analyze the user's query and determine what specific information you need.
2. Plan your search strategy (you can perform up to 3 targeted searches).
3. Use the web_search function to gather relevant, current information.
4. Synthesize the search results into a detailed, narrative report.
5. Leave *Two* lines after each paragraph.

IMPORTANT: Your main response should be a narrative report to the User. Synthesize the information from web searches into this report. 
A separate 'Sources' section, listing the URLs and titles of the web pages used, will be automatically generated and appended to your response by the system after your narrative. 
**Therefore, do not attempt to list or format the sources yourself within the main body of your report.** You can, however, mention that information was retrieved from web searches if it flows naturally within your narrative.

Start by performing your first search with the most relevant query terms for this topic."""

        # Send initial message with tools
        response = chat.send_message(
            initial_prompt,
            tools=[{"function_declarations": [web_search_declaration]}]
        )
        print(f"Initial Gemini response received")

        print(f"=== STEP 3: Response Processing Loop ===")
        iteration_count = 0
        max_iterations = 3  
        
        while iteration_count < max_iterations:
            print(f"--- Iteration {iteration_count + 1} ---")
            
            # Check all parts for function calls
            function_calls_found = []
            text_parts = []
            
            for i, part in enumerate(response.parts):
                print(f"Part {i}: {type(part).__name__}")
                
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls_found.append(part.function_call)
                    print(f"Function call found: {part.function_call.name}")
                
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                    print(f"Text part found: {part.text[:100]}...")

            # Process function calls
            if function_calls_found:
                print(f"Processing {len(function_calls_found)} function calls")
                
                # Handle the first function call
                function_call = function_calls_found[0]
                
                if function_call.name == "web_search":
                    search_query = function_call.args["query"]
                    print(f"Executing Gemini-requested search: {search_query}")
                    
                    # Perform the search (now returns dict with content and sources)
                    search_result = web_search(search_query)
                    search_content = search_result["content"]
                    search_sources = search_result["sources"]
                    
                    print(f"Search results: {search_content}")
                    print(f"Sources found: {len(search_sources)}")
                    
                    # Add sources to the collection
                    all_sources.extend(search_sources)
                    
                    thinking_steps.append(f"Step {3 + iteration_count} - Gemini searched '{search_query}': {search_content}")
                    
                    # Send function response back to Gemini
                    function_response_parts = [
                        {
                            "function_response": {
                                "name": "web_search",
                                "response": {
                                    "content": search_content
                                }
                            }
                        }
                    ]
                    
                    print("Sending function response back to Gemini")
                    response = chat.send_message(
                        function_response_parts,
                        tools=[{"function_declarations": [web_search_declaration]}]
                    )
                    print("Received response after function call")
                    iteration_count += 1
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown function call: {function_call.name}")
            
            else:
                # No function calls found - this should be the final response
                print("No function calls found - processing final response")
                
                if text_parts:
                    final_response = "\n".join(text_parts)
                    print(f"Final response assembled: {len(final_response)} characters")
                else:
                    # Try to get response.text directly
                    try:
                        final_response = response.text if hasattr(response, 'text') else 'No response generated'
                    except:
                        final_response = 'Response processing error'
                    print(f"Using direct response.text: {len(final_response)} characters")
                
                # Add sources section to the final response
                if all_sources:
                    sources_section = "\n\n\n## Sources\n\n"
                    for i, source in enumerate(all_sources, 1):
                        title = source.get('title', 'Unknown Title')
                        url = source.get('url', '')
                        
                        if url:
                            sources_section += f"{i}.  [{title}]({url})\n"
                        else:
                            sources_section += f"{i}.  {title}\n"
                    
                    final_response += sources_section
                    print(f"Added {len(all_sources)} sources to response")
                else:
                    print("No sources found to add")
                
                thinking_process = "\n".join(thinking_steps)
                
                print(f"=== STEP 4: Response Assembly ===")
                print(f"Thinking process steps: {len(thinking_steps)}")
                print(f"Final response length: {len(final_response)}")
                
                # Update history with new query and response
                updated_history = request.history + [
                    Message(role="user", content=request.query),
                    Message(role="bot", content=final_response)
                ]
                
                print(f"=== STEP 5: Return Results ===")
                return {
                    "response": final_response, 
                    "thinking": thinking_process, 
                    "history": updated_history
                }
        
        # If we reach here, max iterations were reached
        print(f"=== ERROR: Max iterations reached ===")
        print(f"Last response parts: {[type(part).__name__ for part in response.parts]}")
        
        # Try to extract any text from the last response
        final_text_parts = []
        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                final_text_parts.append(part.text)
        
        if final_text_parts:
            fallback_response = "\n".join(final_text_parts)
            
            # Add sources section to fallback response
            if all_sources:
                sources_section = "\n\n\n## **Sources**\n\n"
                for i, source in enumerate(all_sources, 1):
                    source_type = source.get('type', 'Web')
                    title = source.get('title', 'Unknown Title')
                    url = source.get('url', '')
                    
                    if url:
                        sources_section += f"{i}. **{source_type}**\t [{title}]({url})\n"
                    else:
                        sources_section += f"{i}. **{source_type}**\t {title}\n"
                
                fallback_response += sources_section
            
            thinking_process = "\n".join(thinking_steps + ["Note: Max iterations reached, returning partial response"])
            
            updated_history = request.history + [
                Message(role="user", content=request.query),
                Message(role="bot", content=fallback_response)
            ]
            
            return {
                "response": fallback_response, 
                "thinking": thinking_process, 
                "history": updated_history
            }
        
        raise HTTPException(status_code=500, detail="Max iterations reached without usable response")
        
    except Exception as e:
        print(f"=== ERROR in deep_search ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# Enhanced web search function with source tracking
def web_search(query: str) -> dict:
    try:
        print(f"Executing SerpAPI search for: {query}")
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 5,  # Explicitly request 5 results
            "location": "India",  # Add location for better results
        })
        result = search.get_dict()
        
        # Debug: Print what SerpAPI returns
        print(f"SerpAPI response keys: {list(result.keys())}")
        
        # Check for errors in the response
        if 'error' in result:
            print(f"SerpAPI Error: {result['error']}")
            return {
                "content": f"Search API Error: {result['error']}",
                "sources": []
            }
        
        # Try different result types
        organic_results = result.get('organic_results', [])
        news_results = result.get('news_results', [])
        answer_box = result.get('answer_box', {})
        
        print(f"Organic results: {len(organic_results)}")
        print(f"News results: {len(news_results)}")
        print(f"Answer box: {'Yes' if answer_box else 'No'}")
        
        # Collect snippets and sources
        snippets = []
        sources = []
        
        # Get answer box if available
        if answer_box and answer_box.get('answer'):
            snippets.append(f"• [Featured] {answer_box['answer']}")
            if answer_box.get('link'):
                sources.append({
                    "title": answer_box.get('title', 'Featured Answer'),
                    "url": answer_box['link'],
                })
        
        # Get organic results
        for item in organic_results[:3]:
            if item.get('snippet'):
                snippets.append(f"• {item['snippet']}")
                sources.append({
                    "title": item.get('title', 'Unknown Title'),
                    "url": item.get('link', ''),
                })
        
        # Get news results
        for item in news_results[:2]:
            if item.get('snippet'):
                snippets.append(f"• [News] {item['snippet']}")
                sources.append({
                    "title": item.get('title', 'Unknown News Title'),
                    "url": item.get('link', ''),
                })
        
        if snippets:
            return {
                "content": "\n".join(snippets),
                "sources": sources
            }
        else:
            # If still no results, return debug information
            return {
                "content": f"No results found. API returned {len(result)} keys: {list(result.keys())[:5]}",
                "sources": []
            }
            
    except Exception as e:
        print(f"SerpAPI Exception: {str(e)}")
        return {
            "content": f"Search failed with error: {str(e)}",
            "sources": []
        }
