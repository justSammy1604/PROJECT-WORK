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
from langchain_community.vectorstores import Chroma
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
    model="command",  # change the model
    temperature=0.4
)

embedding_model = CohereEmbeddings(
    cohere_api_key=api_key,
    model_name="large" # Or "small", "multilingual-22-12", etc. - choose your Cohere embedding model
)

class RedisSemanticCache:
    def __init__(self, 
                 host='localhost', 
                 port=6379, 
                 db=0, 
                 similarity_threshold=0.85, 
                 max_entries=20,
                 ttl=3600):
        self.client = redis.Redis(host=host, port=port, db=db)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl  # In seconds
        self.access_zset = 'cache_access'
        self.data_hash = 'cache_data'
        self.frequency_hash = 'cache_frequency'

    def cosine_similarity(self, vec_1, vec_2):
        a = np.array(vec_1)
        b = np.array(vec_2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def current_time(self):
        return int(datetime.now().timestamp() * 1000)

    def query_semantic_cache(self, query_embedding, threshold=0.8):
        """Enhanced cache query with TTL and frequency awareness"""
        now = self.current_time()
        ttl_cutoff = now - (self.ttl * 1000)
        
        # Cleanup expired entries first
        expired = self.client.zrangebyscore(self.access_zset, 0, ttl_cutoff)
        if expired:
            self.client.zrem(self.access_zset, *expired)
            self.client.hdel(self.data_hash, *expired)
            self.client.hdel(self.frequency_hash, *expired)

        best_match = None
        high_score = -1
        
        for key, value in self.client.hscan_iter(self.data_hash):
            entry = json.loads(value)
            similarity = self.cosine_similarity(query_embedding, entry['embedding'])
            
            if similarity > high_score and similarity > threshold:
                high_score = similarity
                best_match = {
                    'response': entry['response'],
                    'key': key.decode(),
                    'frequency': int(self.client.hget(self.frequency_hash, key) or 0)
                }

        if best_match:
            # Update frequency and access time
            pipeline = self.client.pipeline()
            pipeline.hincrby(self.frequency_hash, best_match['key'], 1)
            pipeline.zadd(self.access_zset, {best_match['key']: now})
            pipeline.execute()
            return best_match['response']
        
        return None

    def add_entry(self, embedding, response):
        """Add new entry with automatic eviction management"""
        entry_id = f"entry_{self.client.incr('cache_counter')}"
        now = self.current_time()
        
        # Store data
        self.client.hset(self.data_hash, entry_id, json.dumps({
            'embedding': embedding,
            'response': response
        }))
        
        # Initialize frequency and access time
        pipeline = self.client.pipeline()
        pipeline.hset(self.frequency_hash, entry_id, 1)
        pipeline.zadd(self.access_zset, {entry_id: now})
        
        # Enforce max entries with frequency-aware eviction
        current_count = self.client.zcard(self.access_zset)
        if current_count > self.max_entries:
            # Get all valid entries (within TTL)
            valid_entries = self.client.zrangebyscore(
                self.access_zset, 
                now - (self.ttl * 1000), 
                '+inf',
                withscores=True
            )
            
            # If still over limit after TTL cleanup
            if len(valid_entries) > self.max_entries:
                # Get frequency scores for valid entries
                entries_with_freq = [
                    (entry[0].decode(), int(self.client.hget(self.frequency_hash, entry[0]) or 0))
                    for entry in valid_entries
                ]
                
                # Sort by frequency (ascending) and remove least frequent
                entries_with_freq.sort(key=lambda x: x[1])
                to_remove = entries_with_freq[:len(valid_entries) - self.max_entries]
                
                # Remove from all data structures
                remove_ids = [entry[0] for entry in to_remove]
                pipeline.zrem(self.access_zset, *remove_ids)
                pipeline.hdel(self.data_hash, *remove_ids)
                pipeline.hdel(self.frequency_hash, *remove_ids)
        
        pipeline.execute()

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
    cached_response = cache.get_similar(query_embedding)
    if cached_response:
        print('Cache Hit!!')
        return cached_response
    
    print('Cache Miss!!')
    result = rag_chain({"query": query})
    response = result['result']

    cache.add_entry(query_embedding,response)
    return response
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

