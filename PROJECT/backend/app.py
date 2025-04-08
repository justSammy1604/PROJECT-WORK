import os
import json
import time
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from pypdf import PdfReader
import sys
from unicodedata import category
import spacy

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_MODEL')
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
hf_key = os.getenv('HF_TOKEN')

# Semantic Cache Implementation
def init_cache():
    index = faiss.IndexFlatL2(768)
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", use_auth_token=hf_key)
    return index, encoder


def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache_data = json.load(file)
        cache = {
            "questions": OrderedDict(cache_data.get("questions", {})),
            "response_text": cache_data.get("response_text", []),
            "frequencies": cache_data.get("frequencies", {})
        }
    except FileNotFoundError:
        cache = {"questions": OrderedDict(), "response_text": [], "frequencies": {}}
    return cache


def store_cache(json_file, cache):
    print(f"Saving to {json_file}")
    with open(json_file, "w") as file:
        json.dump(cache, file)


class SemanticCache:
    def __init__(self, json_file="cache_file.json", threshold=0.35, max_response=100):
        self.index, self.encoder = init_cache()
        self.threshold = threshold
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.question_addition_index = {q: i for i, q in enumerate(self.cache["questions"].keys())}
        if self.cache["questions"]:
            embeddings = self.encoder.encode(list(self.cache["questions"].keys()))
            self.index.add(embeddings)

    def evict(self):
        while len(self.cache["questions"]) > self.max_response:
            self.cache["questions"].popitem(last=False)
            self.cache["response_text"].pop(0)

    def ask(self, question: str, rag_chain) -> str:
        start_time = time.time()
        embedding = self.encoder.encode([question])

        D, I = self.index.search(embedding, 1)
        if len(I[0]) > 0 and I[0][0] >= 0 and D[0][0] <= self.threshold:
            row_id = int(I[0][0])
            response_text = self.cache["response_text"][row_id]
            matched_question = list(self.cache["questions"].keys())[row_id]
            self.cache["frequencies"][matched_question] = self.cache["frequencies"].get(matched_question, 0) + 1
            self.cache["questions"].move_to_end(matched_question)
            print("Response came from CACHE.")
        else:
            response_text = query_response(question, rag_chain)  # Use the existing query_response function
            self.cache["frequencies"][question] = 1
            self.question_addition_index[question] = len(self.cache["response_text"])
            self.cache["questions"][question] = None
            self.cache["response_text"].append(response_text)
            self.index.add(embedding)
            print("Response came from VECTORSTORE/RAG.")

        self.evict()
        store_cache(self.json_file, self.cache)
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        return response_text

    def cleanup_cache(self):
        low_freq_questions = [(q, self.cache["frequencies"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["frequencies"].get(q, 0) < 2]
        high_freq_questions = [(q, self.cache["frequencies"].get(q, 0)) for q in self.cache["questions"].keys() if self.cache["frequencies"].get(q, 0) >= 2]

        low_freq_questions.sort(key=lambda x: list(self.cache["questions"].keys()).index(x[0]))
        kept_questions = [q for q, freq in high_freq_questions]

        if not kept_questions:
            self.cache["questions"] = OrderedDict()
            self.cache["response_text"] = []
            self.cache["frequencies"] = {}
        else:
            new_questions = OrderedDict()
            new_response_text = []
            new_frequencies = {}
            for q in kept_questions:
                new_questions[q] = None
                new_response_text.append(self.cache["response_text"][self.question_addition_index[q]])
                new_frequencies[q] = self.cache["frequencies"][q]
            self.cache["questions"] = new_questions
            self.cache["response_text"] = new_response_text
            self.cache["frequencies"] = new_frequencies

        store_cache(self.json_file, self.cache)
        print("Cache cleaned up. Low-frequency questions (< 2) removed, frequencies preserved in cache_file.json.")


# Original LangChain Functions
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
  if(True):  
    template = """You are a highly skilled and professional financial advisor. Your role is to provide accurate, clear, 
  and concise financial advice solely based on the information provided in the given data source. 
  Do not make assumptions or include any information not explicitly stated in the source.
  If a question is beyond the scope of the data, politely respond with: "I'm sorry, but I can only provide information based on the given data source."
  Always ensure that your responses are in a professional and respectful tone, and provide actionable insights where possible based on the user's query and the available data.

  *Output Format:*.

1.  *answer (Required):* A string containing the textual financial advice/analysis .
2.  *graphData (Optional):* Include this object only if a visual representation significantly clarifies trends, comparisons, or proportions in the data.
    *   type (Required if graphData is present): String - must be 'line', 'bar', or 'pie'.
        *   'line': Use for trends over time.
        *   'bar': Use for comparing distinct categories/values.
        *   'pie': Use for showing parts of a whole (proportions, allocation %).
    *   data (Required if graphData is present): An array of objects. Each object *must* have:
        *   name: (String) Label, category, or time point.
        *   value: (Number) The corresponding numerical value.

*Constraint:* Generate graphData only when it adds substantial value beyond the text answer. Ensure all graphData content is related to the source data.

*Example Structure (Line):*
{
  "answer": "Analysis based on data...",
  "graphData": {
    "type": "line",
    "data": [ { "name": "Year1", "value": 100 }, { "name": "Year2", "value": 120 } ]
  }
}
*Example Structure (Bar):*
{
  "answer": "Comparison based on data...",
  "graphData": {
    "type": "bar",
    "data": [ { "name": "CategoryA", "value": 50 }, { "name": "CategoryB", "value": 75 } ]
  }
}
*Example Structure (Pie):*
{
  "answer": "Allocation breakdown...",
  "graphData": {
    "type": "pie",
    "data": [ { "name": "AssetX", "value": 60 }, { "name": "AssetY", "value": 40 } ]
  }
}
*Example Structure (Text Only):*
{
  "answer": "This query can be answered with text alone, based on the data..."
}

        context: {context} ,
        Question: {question} ,
        Helpful Answer:"""
    
    template = """Answer the questions to the best of your ability, you have the ability to display graphs using graphData.
         *Output Format:*.
1.  *answer (Required):* A string containing the textual financial advice/analysis .
2.  *graphData (Optional):* Include this object only if a visual representation significantly clarifies trends, comparisons, or proportions in the data.
    *   type (Required if graphData is present): String - must be 'line', 'bar', or 'pie'.
        *   'line': Use for trends over time.
        *   'bar': Use for comparing distinct categories/values.
        *   'pie': Use for showing parts of a whole (proportions, allocation %).
    *   data (Required if graphData is present): An array of objects. Each object *must* have:
        *   name: (String) Label, category, or time point.
        *   value: (Number) The corresponding numerical value.

*Constraint:* Generate graphData only when it adds substantial value beyond the text answer.
*Example Structure (Line):*
{
  "answer": "Analysis based on data...",
  "graphData": {
    "type": "line",
    "data": [ { "name": "Year1", "value": 100 }, { "name": "Year2", "value": 120 } ]
  }
}
*Example Structure (Bar):*
{
  "answer": "Comparison based on data...",
  "graphData": {
    "type": "bar",
    "data": [ { "name": "CategoryA", "value": 50 }, { "name": "CategoryB", "value": 75 } ]
  }
}
*Example Structure (Pie):*
{
  "answer": "Allocation breakdown...",
  "graphData": {
    "type": "pie",
    "data": [ { "name": "AssetX", "value": 60 }, { "name": "AssetY", "value": 40 } ]
  }
}
*Example Structure (Text Only):*
{
  "answer": "This query can be answered with text alone, based on the data..."
}

        context: {context} ,
        Question: {question} ,
        Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='result',
        return_messages=True
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
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
    except Exception as e:
        return f"An error occurred: {str(e)}"


def load_and_process(doc_source):
    all_docs = []
    loader = DirectoryLoader(doc_source, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()
    all_docs.extend(documents)
    return split_text(sw_rem_main(all_docs))


def rag_pipeline(document_sources, useData):
    processed_docs = load_and_process(document_sources)
    vector_store = vectordb_information(processed_docs)
    rag_chain = rag_model(vector_store, useData)
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
    documents = []
    nlp = spacy.load("en_core_web_sm")
    codepoints = range(sys.maxunicode + 1)
    punctuation = {c for i in codepoints if category(c := chr(i)).startswith("P")} | {"\n", " "}
    
    for text in texts:
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        for i in punctuation:
            for token in filtered_tokens.copy():
                if i in token:
                    filtered_tokens.remove(token)
        documents.append(filtered_tokens)
    return documents