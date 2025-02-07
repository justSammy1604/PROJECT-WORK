from langchain_cohere import ChatCohere, CohereEmbeddings
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

data = 'pdfdata'

load_dotenv()

api_key = os.getenv('COHERE_API_KEY')
model = ChatCohere(model='command-r-plus',cohere_api_key=api_key, temperature=0.4, convert_system_message_to_human=True)
embedding_model = CohereEmbeddings(model="embed-multilingual-v2.0",cohere_api_key=api_key)

loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_split.split_documents(all_docs)

vectorstore = Chroma.from_documents (documents=docs, embedding=embedding_model, persist_directory="./chroma_db2")

retriever = vectorstore.as_retriever()

prompt_template = """
You are a highly skilled and professional financial advisor. Your role is to provide accurate, clear, 
  and concise financial advice solely based on the information provided in the given data source. 
  Do not make assumptions or include any information not explicitly stated in the source.
  If a question is beyond the scope of the data, politely respond with: "I'm sorry, but I can only provide information based on the given data source."
  Always ensure that your responses are in a professional and respectful tone, and provide actionable insights where possible based on the user's query and the available data.
  {context}
  Question: {question}
  Helpful Answer:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

questions = ["What did the president say about Justice Breyer?", 
             "What did the president say about Intel's CEO?",
             "What did the president say about gun violence?",
            ]
ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                ["The president asked Congress to pass proven measures to reduce gun violence."]]
answers = []
contexts = []

# Inference
for query in questions:
  answers.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
print(df)