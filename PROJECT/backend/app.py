#To add main Langchain code here.
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

