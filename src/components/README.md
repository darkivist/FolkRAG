# Components
Main components for this RAG system consist of the `retriever.py` and `generator.py`, and `create_vectorstore.py`. 
- [Usage](#usage)
- [Retriever Details](#retriever-details)
- [Generator Details](#generator-details)

# Retriever Details
The retriever module for information retrieval from vector stores using embedding models. The retriever leverages AWS Bedrock, DeepLake, and LangChain for advanced query processing and document search functionalities. Primary features include:
- **Embedding Models**: Supports embedding models `instructor-xl` and Amazon Bedrock.
- **Vector Store**: Uses DeepLake for vector storage and retrieval.
- **HyDE Generator**: Generates hypothetical document embeddings for better query matching.
- **TF-IDF-based Reranking**: Improves retrieval accuracy by reranking results based on term relevance, freshness, and keyword coverage.

### Vector Store
A vector store is selected by the user and loaded in by the application. There are five options available to choose from:
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 250 on **all** of the data. This is not widely available due to size.
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 250 on **all** of the data. This is not widely available due to size.
* Vector store constructed using `amazon/titan` embeddings and chunk sizes of 250 on **all** of the data.This is not widely available due to size.
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 250 on a **small sample** of the data. This is widely available on the git repo.
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 1,000 on a **small sample**  of the data. This is widely available on the git repo.

### HyDE Generator
The `hyde_generator` function uses AWS Bedrock's Claude model to generate and process variations of the users' query and their responses. The LLM generates an initial response for the original query, rewrites the query into slightly different variations, and generates answers for each variation. Each query-and-response is generated using progressively higher temperatures (i.e., 0.7, 0.8, 0.9) to balance coherence and creativity while ensuring variety. Finally, each query-and-response is combined into a single string, or "document", for a total of three hypothetical documents that will each be searched against the vector store. 

Query generation prompt example:
```
Rewrite this query to be slightly different but similar in meaning: {query}
```

Response generation prompt example:
```
You are a document that answers this question {query}.

Write a short, natural paragraph that directly answers this question. Include additional relevant information if possible.
```

### Document Retrieval
The `test_document_retrieval` function receives each generated "document" containing the query-and-response and searches the vector store to retrieve `top_k` matching documents for each. Given three hypothetical documents are generated for retrieval, a `top_k` of two, for example, would yield a combined six fetched documents. All of the retrieved documents are then combined and de-duplicated.

### TF-IDF Reranker
The `rerank_with_custom_tfidf` extracts the metadata and the content from the retrieved documents and calculates relevance using a combination of TF-IDF cosine similarity, document freshness (favoring newer documents), and keyword coverage from the query. These scores are weighted (50% TF-IDF, 30% freshness, 20% keyword coverage) to generate a total score for each document. The `top_k` top-ranked documents are returned along with their filenames, prioritizing the most relevant and up-to-date information for text generation augmentation.

# Generator Details
The `generator` class processes user queries by integrating context and metadata, formulates well-structured prompts for Claude, and invokes AWS Bedrock to generate precise and relevant responses. The system is built to work with document collections, retrieving the best matching content and associated metadata for answer generation. Primary features include:
1. **Prompt Generation**: Formats structured prompts with query, context, and metadata for Amazon's Claude LLM.
2. **Metadata Integration**: Dynamically incorporates metadata from the most relevant document (highest scoring during the aforementioned reranking step) to enhance response specificity.
3. **AWS Bedrock Integration**: Utilizes Bedrock's Claude models for response generation.

### Prompt Example
```
Human: Please answer the following query based on the provided context and metadata.
Query: What are the key findings from the 2024 climate study?
Context: [Relevant text from top-ranked documents]
Metadata: [Relevant metadata from top-ranked document]

Instructions: 
1. Answer the question using ONLY the information provided in the Context and Metadata above.
2. Do NOT include any information that is not explicitly stated in the Context or Metadata.
3. Begin your answer with a direct response to the question asked.
4. Include relevant details from the Context and Metadata to support your answer.
5. Pay special attention to the recording date, contributors, and locations provided in the metadata.
6. Inform the user of what document filename they can find the information in.

Your Answer here:
```
# Create Vector Store Details
### Required Data Structure
Your data directory must follow this structure:

```
data/
├── txt/              # Plain text documents
├── transcripts/      # Text transcriptions
├── pdf/
│   └── txtConversion/  # OCR-converted PDF text
└── loc_dot_gov_data/   # Metadata directory
    └── {collection_name}/
        ├── file_list.csv
        └── search_results.csv
```

### Important Notes
* All text files must be UTF-8 encoded
* File names should follow AFC identifier pattern (e.g., afc2021007_002_ms01.txt)
* Transcripts should be named as {original_name}_en.txt or {original_name}_en_translation.txt
* Each collection needs file_list.csv and search_results.csv with proper metadata

# License
This project is licensed under the MIT License. See the LICENSE file in the root directory for more information.
