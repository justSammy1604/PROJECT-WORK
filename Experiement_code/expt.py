import os
import time
from dotenv import load_dotenv 
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain.chains import RetrievalQA 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
import evaluate 

# Load environment variables
load_dotenv()
api_key = os.getenv('COHERE_API_KEY') 

# Initialize model and embedding
model = ChatCohere(model='command-r-plus', cohere_api_key=api_key, temperature=0.4, convert_system_message_to_human=True)
embedding_model = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=api_key)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True) 
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key) 

# Load and process documents
data = 'pdfdata'
all_docs = []
loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
all_docs.extend(documents)
text_split = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_split.split_documents(all_docs)

# Initialize vector database
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory="./chroma_db")
vectorstore.persist()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Define Prompt
prompt_template = """
You are a highly skilled and professional financial advisor. Your role is to provide accurate, clear, 
and concise financial advice solely based on the information provided in the given data sources. 
Do not make assumptions or include any information not explicitly stated in the source.
If a question is beyond the scope of the data, politely respond with: "I'm sorry, but I can only provide information based on the given data source."
Always ensure that your responses are in a professional and respectful tone, and provide actionable insights where possible based on the user's query and the available data.
{context}
Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(model, retriever=retriever, return_source_documents=True)

# Function to retrieve answers
def get_cached_query(query):
    result = qa_chain({"query": query})
    return result['result']

query = "What is a mutual fund, and how does it work?"
retrieved_docs = retriever.get_relevant_documents(query)
generated_answer = get_cached_query(query)

# Measure Time Taken
start = time.time()
response = get_cached_query(query)
end = time.time()
print(f"\nElapsed Time: {(end-start):.4f} seconds")

# Evaluation Metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

# Perform evaluation with respect to LLM
predictions = [generated_answer]
references = [[doc.page_content for doc in retrieved_docs]]  # Using LLM's own response for reference

# Compute BLEU, ROUGE, METEOR scores
bleu_score = bleu_metric.compute(predictions=predictions, references=references)
rouge_score = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references])
meteor_score = meteor_metric.compute(predictions=predictions, references=[ref[0] for ref in references])

# Print Evaluation Scores
print("BLEU Score:", bleu_score)
print("ROUGE Score:", rouge_score)
print("METEOR Score:", meteor_score)

