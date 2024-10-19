import os
import sys
import time
import torch
import gc
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.rag_generator_deeplake import RAGGenerator
from component.rag_pipeline_deeplake import RAGPipeline
from component.metadata_processor import process_metadata


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def chunk_documents(documents: List[Document], chunk_size: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,
        length_function=len,
    )

    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_doc = Document(
                page_content=chunk,
                metadata={**doc.metadata, 'chunk_id': i}
            )
            chunked_documents.append(chunked_doc)
    return chunked_documents

def retriever_eval():
    set_seed(42)
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    chunk_size = 100  # Fixed chunk size of 100
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'

    metadata = process_metadata(data_dir)
    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor')
        
    if delete_existing:
        text_retriever.delete_dataset()

    documents = text_retriever.load_data(data_dir, metadata)
    
    if len(documents) == 0:
        print("No documents loaded. Check the load_data method in RAGRetriever.")
        return

    chunked_documents = chunk_documents(documents, chunk_size)
    num_chunks = len(chunked_documents)
    if num_chunks == 0:
        print("No chunks created. Check the chunking process.")
        return

    # Generate embeddings if the dataset is empty
    if text_retriever.is_empty():
        print("Generating embeddings for chunked documents...")
        text_retriever.generate_embeddings(chunked_documents)
        print("Embeddings generated and saved.")
    else:
        print("Using existing embeddings.")

    queries_answers = [
        ("Complete this sentence: 'The mules are not hungry. They're lively and'", "sr22a_en.txt", "gay"),
        ("Complete this sentence: 'Take a trip on the canal if you want to have'", "sr28a_en.txt or sr13a_en.txt", "fun"),
        ("What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?", "sr02b_en.txt", "Barbrae Allen"), 
        ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "sr28a_en.txt", "Barbara Allen"),
        ("Complete this phrase from the gospel train song: 'The gospel train is'", "sr26a_en.txt", "night"), 
        ("In the song 'Barbara Allen,' where was Barbara Allen from?", "sr02b_en.txt", "Scarlett town"),
        ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "sr08a_en.txt", "A year or two or three at most"),
        ("What instrument does Captain Nye mention loving?", "sr22a_en.txt", "old fiddled mouth organ banjo"), 
        ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "sr27b_en.txt", "whiskers"),
        ("Complete this line from a song: 'We land this war down by the'", "sr05a_en.txt", "river"),
        ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "sr01b_en.txt", "Marry/Mary at all"),
        ("What does the song say will 'outshine the sun'?", "sr17b_en.txt", "We'll/not"),
        ("In the 'Dying Cowboy' song, where was the cowboy born?", "sr20b_en.txt", "Boston")
    ]

    # Setup empty dataframe
    df_results = pd.DataFrame(columns=["Query", "Expected Doc", "Retrieved Doc", "Expected Answer", "Retrieved Content", "Match"])

    # Iterate over each item, retrieve documents, and add to the DataFrame
    for query_info in queries_answers:
        query, expected_doc, expected_answer = query_info
        query_result, original_filename, document_content = text_retriever.test_document_retrieval(query)
        match = (original_filename == expected_doc)
        new_row = pd.DataFrame({
            "Query": [query_result],
            "Expected Doc": [expected_doc],
            "Retrieved Doc": [original_filename],
            "Expected Answer": [expected_answer],
            "Retrieved Content": [document_content],
            "Match": [match]
        })
        df_results = pd.concat([df_results, new_row], ignore_index=True)

    # Get csv file of results
    csv_path = os.path.join(src_dir, 'query_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"The DataFrame has been saved to '{csv_path}'.")


def main():
    set_seed(42)

    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    chunk_size = 100  # Fixed chunk size of 100

    print(f"Data directory: {data_dir}")
    print(f"Chunk size: {chunk_size}")

    # Ask if user wants to delete existing data
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'

    # Initialize components
    metadata = process_metadata(data_dir)
    print(f"Number of documents with metadata: {len(metadata)}")

    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    print(f"Dataset path: {dataset_path}")

    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor')
    print("RAGRetriever initialized")

    if delete_existing:
        text_retriever.delete_dataset()
        print("Existing dataset deleted.")

    # Load and prepare documents
    documents = text_retriever.load_data(data_dir, metadata)
    print(f"Number of documents loaded: {len(documents)}")

    if len(documents) == 0:
        print("No documents loaded. Check the load_data method in RAGRetriever.")
        return

    print("Sample of loaded documents:")
    for i, doc in enumerate(documents[:3]):  # Print details of first 3 documents
        print(f"Document {i+1}:")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("---")

    chunked_documents = chunk_documents(documents, chunk_size)
    num_chunks = len(chunked_documents)
    print(f"Prepared {num_chunks} chunks with size {chunk_size}")

    if num_chunks == 0:
        print("No chunks created. Check the chunking process.")
        return

    print("Sample of chunked documents:")
    for i, chunk in enumerate(chunked_documents[:3]):  # Print details of first 3 chunks
        print(f"Chunk {i+1}:")
        print(f"Content preview: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("---")

    # Generate embeddings if the dataset is empty
    if text_retriever.is_empty():
        print("Generating embeddings for chunked documents...")
        text_retriever.generate_embeddings(chunked_documents)
        print("Embeddings generated and saved.")
    else:
        print("Using existing embeddings.")

    # Initialize RAG components
    qa_generator = RAGGenerator(model_name='llama3')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        start_time = time.time()
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query)
        end_time = time.time()

        print("\n--- Results ---")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Number of retrieved documents: {len(retrieved_docs)}")
        print("Response:")
        print(response)
        print("-------------------\n")

    # Cleanup
    del text_retriever, qa_generator, rag_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    #main()
    retriever_eval()