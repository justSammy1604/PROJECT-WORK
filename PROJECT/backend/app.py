#To add main Langchain code here.
import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 

#Have to discuss which vectorDB and model y'all would prefer.

load_dotenv() 
api_key = os.getenv('GOOGLE_API_MODEL') 

model = ChatGoogleGenerativeAI(model='',google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 

