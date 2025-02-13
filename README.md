#### IMPLEMENTATION OF PROJECT   
We'll start implementing the project on the 6th if that's alright with everyone 

We'll follow the flowchart that we showed during our review.
For now, we'll create a RAG chatbot with a few additional features of our own that include:
- Joseph's method of using NLP to extract keywords and based on that generate the possible output.  To be used for the text and query.
- Use a web crawler for data, that is done by Terrence.  

I have already created the respective files to work on, so fork the repos and work on the respective folders assigned to you.
- `Christie` and `Mathias` y'all can work on the frontend. Lemme know which technology y'all will be using for the frontend.
- `Joseph`, `Terrence` and I will work on backend. Y'all lemme know which model, vector db and embedding y'all would prefer. 
- I may alternate `Terrence` on both sides if frontend peeps need any help. 

### NEXT PHASE OF PROJECT
- `Terrence` has to fix the structure of data extracted from the crawler
- `Joseph` has to add the stop word removal feature
- `Mathias` thought of some graphs, have to look into that.
- For the cache, thought of using Redis as the external cache, if y'all find any other ones, lemme know.
- Once the app done, we have to test it thoroughly
- Writing of research paper, ig Sir will tell this on Monday.
- Also, are we creating the app for the public, if so, we will have to create it keeping in mind load testing and all that.
  
 
![SYSTEM FLOW](diagram-export-4-12-2024-11_57_55-pm.png) 


![ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge)
![METEOR](https://huggingface.co/spaces/evaluate-metric/meteor)
![NDCG & MRR](https://blog.stackademic.com/ndcg-vs-mrr-ranking-metrics-for-information-retrieval-in-rags-2061b04298a6)

