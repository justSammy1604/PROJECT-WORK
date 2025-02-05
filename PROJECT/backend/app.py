#To add main Langchain code here. 
import os 
import json
import numpy as np
from datetime import datetime 
import redis 
from dotenv import load_dotenv 
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain.chains.question_answering import load_qa_chain 
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma, Redis
from langchain.chains import RetrievalQA 

from pypdf import PdfReader
import sys 
from unicodedata import category
import spacy

load_dotenv() 
# api_key = os.getenv('GOOGLE_API_MODEL') 
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key) 

api_key = os.getenv('COHERE_API_KEY')

model = ChatCohere(
    cohere_api_key=api_key,
    model="command-r-plus",  # change the model
    temperature=0.4,
    convert_system_message_to_human=True
)

embedding_model = CohereEmbeddings(
    cohere_api_key=api_key,
    model="embed-multilingual-v2.0" # Or "small", "multilingual-22-12", etc. - choose your Cohere embedding model
)

redis_vectorstore = Redis.from_documents(
    docs,
    embedding=embedding_model,
    redis_url="redis://localhost:6379",
    index_name="finance_cache",  # Unique index name for Redis
)

redis_client = redis.Redis(host="localhost", port=6379, db=0)
cache_ttl = 3600

cache = RedisSemanticCache(similarity_threshold=0.62,max_entries=20,ttl=7200)

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
  template = """You are a highly skilled and professional financial advisor. Your role is to provide accurate, clear, 
  and concise financial advice solely based on the information provided in the given data source. 
  Do not make assumptions or include any information not explicitly stated in the source.
  If a question is beyond the scope of the data, politely respond with: "I'm sorry, but I can only provide information based on the given data source."
  Always ensure that your responses are in a professional and respectful tone, and provide actionable insights where possible based on the user's query and the available data.
        {context}
        Question: {question}
        Helpful Answer:"""
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
  qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.similarity_search_with_score(
            search_kwargs={"k": 2}  
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
  return qa_chain

def query_response(query, rag_chain):
  try:
    query_embedding = embedding_model.embed_query(query)
    redis_docs = redis_vectorstore.similarity_search_by_vector(query_embedding, k=1)
    if redis_docs:
        print("Cache Hit!!")
        return redis_docs[0].page_content
    
    print('Cache Miss!!')
    result = rag_chain({"query": query})
    response = result['result']

    redis_vectorstore.add_texts([query], [response])

    #for quick lookup
    redis_client.setex(f"response:{query}", cache_ttl, response)
  except Exception as e:
    return f"An error occurred: {str(e)}"


def load_and_process(doc_source):
  all_docs = []
  for source in doc_source:
    loader = PyPDFLoader(source)  # Example: Replace with the appropriate loader
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

