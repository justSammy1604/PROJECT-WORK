#To add main Langchain code here. 
import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain.chains.question_answering import load_qa_chain 
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA 

from pypdf import PdfReader
import sys
from unicodedata import category
import spacy

load_dotenv() 
api_key = os.getenv('GOOGLE_API_MODEL') 
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 
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
  template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
  qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}  
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
  return qa_chain

def query_response(query, rag_chain):
  try:
    response = rag_chain({"query": query})
    return response['result']
  except Exception as e:
    return f"An error occurred: {str(e)}"


def load_and_process(doc_source):
  all_docs = []
  for source in doc_source:
    loader = PyPDFLoader(source)  # Example: Replace with the appropriate loader
    documents = loader.load()
    all_docs.extend(documents)
      
  return split_text(all_docs)


def rag_pipeline(document_sources):
  processed_docs = load_and_process(document_sources)

  vector_store = vectordb_information(processed_docs)
  
  rag_chain = rag_model(vector_store)

  return rag_chain



#from pypdf import PdfReader
import sys
from unicodedata import category
import spacy



def token_removal(texts):
    #note that texts is all the texts from different documents in an array

    # Open the PDF file
    # importing required modules

    # creating a pdf reader object
    # reader = PdfReader(pdf_path)

    # # printing number of pages in pdf file
    # text =[]
    # # getting a specific page from the pdf file
    # for page in reader.pages:
    #     text.append(page.extract_text())
    documents = []
    nlp = spacy.load("en_core_web_sm")
    codepoints = range(sys.maxunicode + 1)
    punctuation = {c for i in codepoints if category(c := chr(i)).startswith("P")} and {"\n"," "}
    # Loop through each page in the PDF
    for text in texts:
        doc = nlp(text)

        filtered_tokens = [token.text for token in doc if not token.is_stop]


        for i in punctuation:
            for token in filtered_tokens:
                if i in token:
                    filtered_tokens.remove(token)

        documents.append(filtered_tokens)

    return documents


#testing
# content = []
# file = open("/content/Adams_1797.txt", "r")
# content.append(file.read())
# #print(content)
# file.close()

# file = open("/content/Adams_1798.txt", "r")
# content.append(file.read())
# #print(content)
# file.close()

# file = open("/content/Adams_1799.txt", "r")
# content.append(file.read())
# #print(content)
# file.close()

# file = open("/content/Adams_1800.txt", "r")
# content.append(file.read())
# #print(content)
# file.close()
# answer = pdf_to_documents(content)
# for a in answer:
#   print(a)

