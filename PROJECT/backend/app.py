#To add main Langchain code here. 
import os 
import json
import numpy as np
from datetime import datetime 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain.chains.question_answering import load_qa_chain 
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma, Redis
from langchain.chains import RetrievalQA 
from langchain.memory import ConversationBufferMemory
from pypdf import PdfReader
import sys 
from unicodedata import category
import spacy

load_dotenv() 
api_key = os.getenv('GOOGLE_API_MODEL') 
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key) 


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
template = """**System Prompt: Financial Advisor with Optional Graph Output**

**Role:** Financial advisor LLM.
**Core Directive:** Answer user queries accurately and concisely, using **only** the provided data source. Maintain a professional tone. **Never** use external knowledge or make assumptions. If data is insufficient, state so clearly.

**Output Format:** Respond **only** in JSON.

1.  **`answer` (Required):** A string containing the textual financial advice/analysis based *strictly* on the provided data.
2.  **`graphData` (Optional):** Include this object *only if* a visual representation significantly clarifies trends, comparisons, or proportions in the data.
    *   `type` (Required if `graphData` is present): String - must be `'line'`, `'bar'`, or `'pie'`.
        *   `'line'`: Use for trends over time.
        *   `'bar'`: Use for comparing distinct categories/values.
        *   `'pie'`: Use for showing parts of a whole (proportions, allocation %).
    *   `data` (Required if `graphData` is present): An array of objects. Each object **must** have:
        *   `name`: (String) Label, category, or time point.
        *   `value`: (Number) The corresponding numerical value.

**Constraint:** Generate `graphData` *only* when it adds substantial value beyond the text `answer`. Ensure all `graphData` content is directly derived from the source data.

**Example Structure (Line):**
{
  "answer": "Analysis based on data...",
  "graphData": {
    "type": "line",
    "data": [ { "name": "Year1", "value": 100 }, { "name": "Year2", "value": 120 } ]
  }
}
**Example Structure (Bar):**
{
  "answer": "Comparison based on data...",
  "graphData": {
    "type": "bar",
    "data": [ { "name": "CategoryA", "value": 50 }, { "name": "CategoryB", "value": 75 } ]
  }
}
**Example Structure (Pie):**
{
  "answer": "Allocation breakdown...",
  "graphData": {
    "type": "pie",
    "data": [ { "name": "AssetX", "value": 60 }, { "name": "AssetY", "value": 40 } ]
  }
}
**Example Structure (Text Only):**
{
  "answer": "This query can be answered with text alone, based on the data..."
}
Input Data/Context:
{context}

User Question:
{question}

Response (JSON format only):"""

  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
  memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='result',
        return_messages=True
  )
  qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}  
        ),
        return_source_documents=True,
        memory=memory,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
  return qa_chain

def query_response(query, rag_chain):
  try:
    result = rag_chain({"query": query})
    response = result['result']
    return response
    #for quick lookup
  except Exception as e:
    return f"An error occurred: {str(e)}"


def load_and_process(doc_source):
    all_docs = []
    if doc_source == 'data':
      loader = DirectoryLoader(doc_source,glob="*.pdf",show_progress=True,loader_cls=PyPDFLoader)
    elif doc_source == 'crawled_data':
      loader = DirectoryLoader(doc_source, glob="**/*.json", loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.[].content[].main_content', 'text_content':False})
    else:
       raise ValueError("Invalid document source")
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
    #note that texts is all the texts from different documents in an array

    documents = []
    
    nlp = spacy.load("en_core_web_sm")
    codepoints = range(sys.maxunicode + 1)
    punctuation = {c for i in codepoints if category(c := chr(i)).startswith("P")} and {"\n"," "}
    
    for text in texts:
        doc = nlp(text)

        filtered_tokens = [token.text for token in doc if not token.is_stop]


        for i in punctuation:
            for token in filtered_tokens:
                if i in token:
                    filtered_tokens.remove(token)

        documents.append(filtered_tokens)

    return documents

