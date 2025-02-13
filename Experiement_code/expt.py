import os 
import time
import numpy as np
from dotenv import load_dotenv 
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain.chains.question_answering import load_qa_chain 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA 
# from langchain.evalution import QAEvalChain
from fuzzywuzzy import fuzz
data = 'pdfdata'
# data = 'crawled_content.json'
load_dotenv()
# api_key = os.getenv('GOOGLE_API_MODEL')
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=api_key, temperature=0.4, convert_system_message_to_human=True)
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

api_key = os.getenv('COHERE_API_KEY')
model = ChatCohere(model='command-r-plus',cohere_api_key=api_key, temperature=0.4, convert_system_message_to_human=True)
embedding_model = CohereEmbeddings(model="embed-multilingual-v2.0",cohere_api_key=api_key)
# cache = RedisSemanticCache(similarity_threshold=0.85,max_entries=20)


all_docs = []
loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
# loader = JSONLoader(data,jq_schema='.',text_content=False)
documents = loader.load()
all_docs.extend(documents)
text_split = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_split.split_documents(all_docs)

vectorstore = Chroma.from_documents (documents=docs, embedding=embedding_model, persist_directory="./chroma_db")
vectorstore.persist()


retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


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

# **Ground Truth Answers for Evaluation**
ground_truths = {
    "What is a mutual fund, and how does it work?": " A mutual fund is a type of investment vehicle that allows individuals to pool their money together into a single fund, which is then invested in a variety of securities such as stocks, bonds, and other assets.",  
}

# **Manual Implementation of Evaluation Metrics**

# **Recall@K**
def recall_at_k(retrieved_docs, ground_truth, fuzzy_threshold=60):
    """
    Computes Recall@K for retrieval evaluation.
    Args:
        retrieved_docs: List of retrieved documents.
        ground_truth: Ground truth answer or relevant text.
        fuzzy_threshold: Threshold for fuzzy matching (0-100).
    Returns:
        Recall@K score.
    """
    retrieved_texts = [doc.page_content.lower() for doc in retrieved_docs]
    ground_truth = ground_truth.lower()
    
    # Count relevant documents (exact or fuzzy match)
    relevant_docs = 0
    for doc in retrieved_texts:
        if ground_truth in doc:  # Exact match
            relevant_docs += 1
        elif fuzz.partial_ratio(ground_truth, doc) >= fuzzy_threshold:  # Fuzzy match
            relevant_docs += 1
    
    # Compute Recall@K
    return relevant_docs / len(retrieved_texts) if retrieved_texts else 0



def precision_at_k(retrieved_docs, ground_truth):
    """
    Computes Precision@K for retrieval evaluation.
    Args:
        retrieved_docs: List of retrieved documents.
        ground_truth: Ground truth answer or relevant text.
    Returns:
        Precision@K score.
    """
    retrieved_texts = [doc.page_content.lower() for doc in retrieved_docs]
    ground_truth = ground_truth.lower()
    
    # Count relevant documents (exact match)
    relevant_docs = sum(1 for doc in retrieved_texts if ground_truth in doc)
    
    # Compute Precision@K
    return relevant_docs / len(retrieved_texts) if retrieved_texts else 0

# **BLEU Score (Unigram Precision)**
def bleu_score(reference, candidate):
    """Computes BLEU-1 Score (Unigram Precision)."""
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    overlap = sum(1 for token in candidate_tokens if token in reference_tokens)
    precision = overlap / len(candidate_tokens) if candidate_tokens else 0
    return precision

# **ROUGE-L (Longest Common Subsequence)**
def lcs(X, Y):
    """Finds the length of the longest common subsequence (LCS) between two texts."""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]

def ndcg_at_k(retrieved_docs, ground_truth, k=10):
    relevance_scores = [1 if ground_truth.lower() in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
    dcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(relevance_scores))
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal_relevance_scores))
    return dcg / idcg if idcg > 0 else 0

def mrr(retrieved_docs, ground_truth):
    for i, doc in enumerate(retrieved_docs):
        if ground_truth.lower() in doc.page_content.lower():
            return 1 / (i + 1)
    return 0

def rouge_l(reference, candidate):
    """Computes ROUGE-L Score using Longest Common Subsequence (LCS)."""
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    lcs_length = lcs(reference_tokens, candidate_tokens)
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0
    return recall

# **BERTScore (Cosine Similarity of Embeddings)**
def bert_score(reference, candidate, embedding_model):
    """Computes BERTScore using precomputed embeddings."""
    reference_embedding = np.array(embedding_model.embed_query(reference))
    candidate_embedding = np.array(embedding_model.embed_query(candidate))
    return np.dot(reference_embedding, candidate_embedding) / (np.linalg.norm(reference_embedding) * np.linalg.norm(candidate_embedding))

""" def bert_score(reference, candidate, embedding_model):
    Computes BERTScore using cosine similarity of word embeddings.
    reference_embedding = np.array(embedding_model.embed_query(reference))
    candidate_embedding = np.array(embedding_model.embed_query(candidate))
    return (reference_embedding, candidate_embedding) """

def get_cached_query(query):
    result = qa_chain({"query": query})
    response = result['result']

    # cache.add_entry(query_embedding,response)

    return response


query = "What is a mutual fund, and how does it work?"
retrieved_docs = retriever.get_relevant_documents(query)
# for i, doc in enumerate(retrieved_docs):
#     print(f"Retrieved Document {i+1}: {doc.page_content}\n")
generated_answer = get_cached_query(query)
ground_truth = ground_truths.get(query, "")

query_embedding = embedding_model.embed_query(query)

# Fetch embeddings of the first retrieved document
retrieved_embedding = embedding_model.embed_query(retrieved_docs[0].page_content)

# Compare embeddings using cosine similarity
similarity = np.dot(query_embedding, retrieved_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(retrieved_embedding))

# print("Cosine Similarity:", similarity)


# **Evaluate Retrieval & Generation**
found = any(ground_truths["What is a mutual fund, and how does it work?"] in doc.page_content for doc in docs)
print("Ground truth found in dataset:", found)

# recall_k_score = recall_at_k(retrieved_docs, ground_truth)
# precision_k_score = precision_at_k(retrieved_docs, ground_truth)
bleu = bleu_score(ground_truth, generated_answer)
rouge = rouge_l(ground_truth, generated_answer)
# ndcg_k_score = ndcg_at_k(retrieved_docs, ground_truth, k=10)
# mrr_score = mrr(retrieved_docs, ground_truth)
bert = bert_score(ground_truth, generated_answer, embedding_model)

print("\nGenerated Answer:", generated_answer)
# print("\nRetrieval Metrics: Recall@K =", recall_k_score)
# print("Retrieval Metrics: Precision@K =", precision_k_score)
print("\nGeneration Metrics:")
print(f"BLEU Score: {bleu}")
print(f"ROUGE-L Score: {rouge}")
# print(f"NDCG@K: {ndcg_k_score:.4f}")
# print(f"MRR: {mrr_score:.4f}")
print(f"BERTScore: {bert}")

# **Measure Time Taken**
start = time.time()
response = get_cached_query(query)
end = time.time()
# print(f"\nElapsed Time: {(end-start):.4f} seconds")

