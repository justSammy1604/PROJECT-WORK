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
  def __init__(self,host='localhost',port=6379,db=0,similarity_threshold=0.62,max_entries=20):
        self.client=redis.Redis(host=host,port=port,db=db)
        self.similarity_threshold=similarity_threshold
        self.max_entries=max_entries
        self.access_zset='cache_access'
        self.data_hash='cache_data'

    def cosine_similarity(self,vec_1,vec_2):
        a=np.array(vec_1)
        b=np.array(vec_2)
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def current_time(self):
        return int(datetime.now().timestamp()*1000)
    
    def get_similar(self,embedding):
        all_entries = self.client.hgetall(self.data_hash)
        best_match=None
        high_similarity=-1

        for key, value in all_entries.items():
            entry=json.loads(value)
            similarity=self.cosine_similarity(embedding,entry['embedding'])
            if similarity>high_similarity and similarity>self.similarity_threshold:
                high_similarity=similarity
                best_match=entry['response']

                self.client.zadd(self.access_zset,{key:self.current_time()})

        return best_match
    
    def add_entry(self,embedding,response):
        entry_id=f'entry_{self.client.incr('cache_counter')}'

        self.client.hset(self.data_hash,entry_id,json.dumps({'embedding':embedding,'response':response}))

        self.client.zadd(self.access_zset,{entry_id:self.current_time()})

        if self.client.zcard(self.access_zset)>self.max_entries:
            oldest=self.client.zrange(self.access_zset,0,0)
            self.client.zrem(self.access_zset,*oldest)
            self.client.hdel(self.data_hash,*oldest)

cache = RedisSemanticCache(similarity_threshold=0.62,max_entries=20)

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

