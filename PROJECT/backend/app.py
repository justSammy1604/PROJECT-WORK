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
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)




def split_text(documents):  #Will add the required functions and definition to create the RAG model.
  text_split = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
  docs = text_split.split_documents(documents)
  return docs

def vectordb_information(docs): 
      vectorstore = Milvus.from_documents(  # or Zilliz.from_documents
        documents=docs,
        embedding=embedding_model,
        connection_args={
        "uri": "./milvus_demo.db",
        },
    drop_old=True,  # Drop the old Milvus collection if it exists
)
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

def response(query, rag_chain):
    try:
        response = rag_chain.invoke({"query": query})
        return response['result']
    except Exception as e:
        return f"An error occurred: {str(e)}"


def load_and_process(doc_source):
    data = 1 # Terrence, you place your Crawler and the text it extracts here in this var.
    all_docs=[]
    for source in doc_source:
        loader=1 #Use the doc loader methods mentioned by Langchain

    documents=loader.load()
    preprocess_docs = 1 #Joseph, preprocess the docs here via stopword removal and NLP techniques to clean the data

    return split_text(all_docs)


def rag_pipeline(document_sources):
    processed_docs = load_and_process(document_sources)

    vector_store = vectordb_information(processed_docs)

    rag_chain = rag_model(vector_store)

    return rag_chain


