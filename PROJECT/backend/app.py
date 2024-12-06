#To add main Langchain code here.
import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
load_dotenv() 
api_key = os.getenv('GOOGLE_API_MODEL') 
model = ChatGoogleGenerativeAI(model='',google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 



def preprocessing(): # Joseph, perform stopword removal on Terrence's data from the crawler here.
  pass



def rag_model():  #Will add the required functions and definition to create the RAG model.
  pass


def response(): #Adding the response part seperately to be called via the API to the frontend
  pass

data = 1 # Terrence, you place your Crawler and the text it extracts here in this var.
