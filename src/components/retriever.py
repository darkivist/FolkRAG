import os
import datetime
import re
import numpy as np
import pandas as pd
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from typing import Dict, List, Any
from langchain.vectorstores import DeepLake
from langchain_community.vectorstores import DeepLake # last one was depracated
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
import streamlit as st
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings

class retriever():
    def __init__(self, project_root, embeddor_name='instructor', vstor_name='vectorstore_all_250'):
        self.embeddor = embeddor_name
        self.vstore = vstor_name
        self.project_root = project_root

    def load_configuration(self, config_dir):
        load_dotenv(dotenv_path=os.path.join(config_dir, '.env'))
        config_file = os.environ['CONFIG_FILE']
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(f"{config_dir}/{config_file}")
        return config

    def create_bedrock_client(self, config):
        session = boto3.Session(
            aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
            aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
            aws_session_token=config['BedRock_LLM_API']['aws_session_token']
        )
        return session.client("bedrock-runtime", region_name="us-east-1")

    def get_embedding_vectors(self, text, embeddings):
        response = embeddings.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']

    def hyde_generator(self, bedrock_client, user_query):
        results = {}

        try:
            # Generate response for the original query
            original_prompt = (
                f"You are a document that answers this question: {user_query}\n"
                "Write a short, natural paragraph that directly answers this question."
                "Include additional relevant information if it adds value in the answer."
                "Do not include any text other than the final answer."
            )
            original_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": original_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }).encode('utf-8')

            original_response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=original_payload,
                contentType="application/json"
            )
            original_body = json.loads(original_response["body"].read())
            response1 = original_body["content"][0]['text']

            results['Original Query'] = user_query
            results['Response1'] = response1

            # Generate first variation of the query
            variation1_prompt = f"Rewrite this query to be slightly different but similar in meaning: {user_query}"
            variation1_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": variation1_prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.8
            }).encode('utf-8')

            variation1_response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=variation1_payload,
                contentType="application/json"
            )
            variation1_body = json.loads(variation1_response["body"].read())
            query2 = variation1_body["content"][0]['text']

            results['Query2'] = query2

            # Generate response for the first variation
            variation1_prompt_response = (
                f"You are a document that answers this question: {query2}\n"
                "Write a short, natural paragraph that directly answers this question."
                "Include additional relevant information if possible."
            )
            variation1_response_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": variation1_prompt_response}
                ],
                "max_tokens": 200,
                "temperature": 0.8
            }).encode('utf-8')

            variation1_response_body = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=variation1_response_payload,
                contentType="application/json"
            )
            variation1_response_text = json.loads(variation1_response_body["body"].read())["content"][0]['text']
            results['Response2'] = variation1_response_text

            # Generate second variation of the query
            variation2_prompt = f"Rewrite this query to be slightly different but similar in meaning: {query2}"
            variation2_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": variation2_prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.9
            }).encode('utf-8')

            variation2_response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=variation2_payload,
                contentType="application/json"
            )
            variation2_body = json.loads(variation2_response["body"].read())
            query3 = variation2_body["content"][0]['text']

            results['Query3'] = query3

            # Generate response for the second variation
            variation2_prompt_response = (
                f"You are a document that answers this question: {query3}\n"
                "Write a short, natural paragraph that directly answers this question."
                "Include additional relevant information if possible."
            )
            variation2_response_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": variation2_prompt_response}
                ],
                "max_tokens": 200,
                "temperature": 0.9
            }).encode('utf-8')

            variation2_response_body = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=variation2_response_payload,
                contentType="application/json"
            )
            variation2_response_text = json.loads(variation2_response_body["body"].read())["content"][0]['text']
            results['Response3'] = variation2_response_text

        except Exception as e:
            print(f"Error processing query '{user_query}': {str(e)}")
            results['Error'] = str(e)

        hydes = results
        return hydes

    def clean_hyde_docs(self, text, response=False):
        # check if non=response phrases are in responess; replace if so
        if response:
            replace_substrings = ["I apologize, but I don't have", "I'm afraid I don't have any concrete", "I'm sorry, but I don't have any", "I don't have any specific information about", "I don't have any specific document or information", "so could risk reproducing copyrighted material without", "I don't have the full lyrics or details", "I don't have access to any particular source", "doing so may infringe on copyrighted", "I understand you're asking about a song involving", "I understand you're looking for a", "I understand your request to", "I don't have information about", "I can't comment", "I don't have access", "I don't have any particular details", "I don't have any details", "You're inquiring about the", "I understand you're asking", "copyright", "copyrighted", "I didn't receive any document", "I'm not acquainted with the", "You seem to be inquiring about", "I'm unable to rewrite or rephrase", "I wasn't provided with any", "I don't have specific information", "I don't have information", "I lack concrete details about", "I apologize, but I don't have", "I'm afraid I don't have any concrete", "I'm sorry, but I don't have any", "I don't have any specific information about", "I don't have any specific document or information", "so could risk reproducing copyrighted material without", "I don't have the full lyrics or details", "I don't have access to any particular source", "doing so may infringe on copyrighted", "I understand you're asking about a song involving", "I understand you're looking for a", "I understand your request to", "I don't have information about", "I can't comment", "I don't have access", "I don't have any particular details", "I don't have any details", "You're inquiring about the", "I understand you're asking", "copyright", "copyrighted", "I didn't receive any document", "I'm not acquainted with the", "You seem to be inquiring about", "I'm unable to rewrite or rephrase", "I wasn't provided with any", "I don't have specific information", "I don't have information", "I lack concrete details about"]
            for sub in replace_substrings:
                if sub in text:
                    text = "retract"
                    break
                else:
                    text = text.strip()
        else:
            # get text between  paragraphs
            matches = re.findall(r'\n\n(.*?)\n\n', text, re.DOTALL)
            if matches:
                text = "\n\n".join(matches)
            else:
                text = text

            # text to delete from the generated text
            remove_substrings = ["Here's a reworded version that conveys a similar meaning:", "This rephrasing maintains the core question about Johnny Garrison and the exact start date of his business, while using different wording.", "Here's a slightly different version with similar meaning:", "Here's a rewritten version that is slightly different but similar in meaning:", "Here's an alternative way to phrase the question while keeping a similar meaning:", "This version still asks about the specific date Johnny Garrison started his company, but uses different terminology and sentence structure to convey the same", "Here's a rephrased version of the question that maintains a similar meaning:", "Here's a rewritten version with a similar meaning:", "Here's a rephrased version with a similar meaning:", "Here's a slightly different version with a similar meaning:", "Here's a rewrite with a similar meaning:", "Here's a revised version with a similar meaning:", "Here's a similar question with slightly different wording:", "I didn't receive any specific document or text to summarize or quote from, so I'll provide a general response:", "A rephrased version of that question could be:", "This maintains the core", "Here's a slightly different version that maintains a similar meaning:", "Here's a reworded version with a similar meaning:", "Here's a reworded version that maintains a similar meaning:", "Here's a rewritten version that conveys a similar meaning:", "Here's a slightly different version that conveys a similar meaning:", "Here's another variation that maintains a similar meaning:", "This version still focuses on the potential reaction of a chef on a canal boat when offended by a passenger, but uses different phrasing an", "I don't have any specific document or source material to reference for this question. Without more context, I can offer a similar rephrasing that conveys a comparable meaning:", "This rephrasing maintains the core", "This rewrite maintains the core elements of the original sentence:", "This version maintains the core idea of a potential conflict with the cook on a canal boat, but changes the wording and perspective slightly.", "This rephrasing keeps the main concept of facing repercussions for angering someone who prepares meals on a water vessel, while altering the specific words an", "This rephrasing keeps the core idea of a plant [growing from a burial site associated with someone of high status, while", "Here's a rewritten version with similar meaning:", "Here's a rephrased version of the request with a similar meaning:", "Here's a rephrased version of that statement with a similar meaning:", "This rewrite maintains the core elements of the original sentence:", "Here's a completion of that statement:", "This statement suggests that something (unspecified in the given fragment) is becoming more prevalent in a particular area compared to its", "Here's a rephrased version of the prompt with a similar meaning:", "Here's a revised version of the prompt:", "This revised version maintains the core request of finishing a statement about the characteristics of older people in a specific country, but uses different phrasing and synonym", "Here's a reworded version that conveys a similar meaning:", "This rephrasing maintains the core question about Johnny Garrison and the exact start date of his business, while using different wording.", "Here's a slightly different version with similar meaning:", "Here's a rewritten version that is slightly different but similar in meaning:", "Here's an alternative way to phrase the question while keeping a similar meaning:", "This version still asks about the specific date Johnny Garrison started his company, but uses different terminology and sentence structure to convey the same", "Here's a rephrased version of the question that maintains a similar meaning:", "Here's a rewritten version with a similar meaning:", "Here's a rephrased version with a similar meaning:", "Here's a slightly different version with a similar meaning:", "Here's a rewrite with a similar meaning:", "Here's a revised version with a similar meaning:", "Here's a similar question with slightly different wording:", "I didn't receive any specific document or text to summarize or quote from, so I'll provide a general response:", "A rephrased version of that question could be:", "This maintains the core", "Here's a slightly different version that maintains a similar meaning:", "Here's a reworded version with a similar meaning:", "Here's a reworded version that maintains a similar meaning:", "Here's a rewritten version that conveys a similar meaning:", "Here's a slightly different version that conveys a similar meaning:", "Here's another variation that maintains a similar meaning:", "This version still focuses on the potential reaction of a chef on a canal boat when offended by a passenger, but uses different phrasing an", "I don't have any specific document or source material to reference for this question. Without more context, I can offer a similar rephrasing that conveys a comparable meaning:", "This rephrasing maintains the core", "This rewrite maintains the core elements of the original sentence:", "This version maintains the core idea of a potential conflict with the cook on a canal boat, but changes the wording and perspective slightly.", "This rephrasing keeps the main concept of facing repercussions for angering someone who prepares meals on a water vessel, while altering the specific words an", "This rephrasing keeps the core idea of a plant growing from a burial site associated with someone of high status, while", "Here's a rewritten version with similar meaning:", "Here's a rephrased version of the request with a similar meaning:", "Here's a rephrased version of that statement with a similar meaning:", "This rewrite maintains the core elements of the original sentence:", "Here's a completion of that statement:", "This statement suggests that something (unspecified in the given fragment) is becoming more prevalent in a particular area compared to its", "Here's a rephrased version of the prompt with a similar meaning:", "Here's a revised version of the prompt:", "This revised version maintains the core request of finishing a statement about the characteristics of older people in a specific country, but uses different phrasing and synonym"]
            pattern = "|".join(map(re.escape, remove_substrings))
            text = re.sub(pattern, "", text)

            # remove any extra spaces and line breaks from Query2 and Query3
            text = text.strip()

        return text

    def convert_to_string(self, value):
        try:
            if isinstance(value, np.ndarray):
                # Handle numpy string arrays
                if value.dtype.kind in ['U', 'S']:  # Unicode or byte string
                    if value.size == 1:
                        # Single element array
                        return value.item()
                    elif value.size > 0:
                        # Multi-element array - take first element
                        return value.flatten()[0]
                    else:
                        # Empty array
                        return ""
                # Handle other numpy arrays
                return str(value.tolist())
            elif isinstance(value, bytes):
                return value.decode('utf-8')
            elif isinstance(value, (list, dict)):
                return json.dumps(value)
            else:
                return str(value)
        except Exception as e:
            print(f"Error converting value to string: {e}")
            return ""

    def parse_metadata(self, metadata):
        try:
            if isinstance(metadata, np.ndarray):
                if metadata.size == 1:
                    metadata = metadata.item()
                elif metadata.size > 0:
                    metadata = metadata.flatten()[0]
                else:
                    return {}

            if isinstance(metadata, (bytes, np.bytes_)):
                metadata = metadata.decode('utf-8')

            if isinstance(metadata, str):
                try:
                    return json.loads(metadata)
                except json.JSONDecodeError:
                    print("Failed to parse metadata JSON string")
                    return {}

            if isinstance(metadata, dict):
                return {k: self.convert_to_string(v) for k, v in metadata.items()}

            print(f"Unexpected metadata type: {type(metadata)}")
            return {}

        except Exception as e:
            print(f"Error processing metadata: {e}")
            return {}

    def search_vector_store(self, embeddor, query, vectorstore, top_k):
        """Modified search method with detailed metadata debugging"""
        # Access the underlying dataset
        ds = vectorstore.vectorstore.dataset
        print("\n=== VECTOR STORE DEBUG ===")
        print(f"Query: {query}")
        print(f"Available tensors: {list(ds.tensors.keys())}")

        # Get dataset size and validate
        dataset_size = len(ds.embedding)
        print(f"Dataset size: {dataset_size}")

        # Generate query embedding and get similarities
        query_embedding = embeddor.embed_query(query)
        embeddor_numpy = ds.embedding.numpy()
        if len(embeddor_numpy.shape) == 3:
            embeddor_numpy = embeddor_numpy.squeeze(1)

        similarities = np.dot(embeddor_numpy, query_embedding) / (
                np.linalg.norm(embeddor_numpy, axis=1) * np.linalg.norm(query_embedding))

        # Get top k indices
        top_k = min(top_k, dataset_size)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        valid_indices = [idx for idx in top_indices if 0 <= idx < dataset_size]

        print(f"\nProcessing top {len(valid_indices)} results")

        # Process results
        processed_results = []
        for idx in valid_indices:
            try:
                print(f"\n--- Processing Result {idx} ---")

                # Debug raw tensor values
                print("Raw tensor values:")
                for tensor_name in ds.tensors.keys():
                    if tensor_name not in ['text', 'embedding']:
                        try:
                            value = ds[tensor_name][int(idx)].numpy()
                            print(f"{tensor_name}: {value}")
                        except Exception as e:
                            print(f"Error accessing tensor {tensor_name}: {e}")

                # Get metadata
                try:
                    metadata_array = ds.metadata[int(idx)].numpy()
                    print(f"\nRaw metadata array: {metadata_array}")
                    metadata_dict = self.parse_metadata(metadata_array)
                    print(f"Parsed metadata: {json.dumps(metadata_dict, indent=2)}")
                except Exception as e:
                    print(f"Error processing metadata: {e}")
                    metadata_dict = {}

                # Process individual tensor fields
                print("\nProcessing individual tensors:")
                for tensor_name in ds.tensors.keys():
                    if tensor_name not in ['text', 'embedding', 'metadata', 'id']:
                        try:
                            tensor_value = ds[tensor_name][int(idx)].numpy()
                            if tensor_value.size > 0:
                                value = tensor_value.item() if tensor_value.size == 1 else tensor_value[0]
                                if value and str(value).strip():
                                    metadata_dict[tensor_name] = value
                                    print(f"Added {tensor_name}: {value}")
                        except Exception as e:
                            print(f"Error processing tensor {tensor_name}: {e}")

                # Add similarity score
                metadata_dict['similarity_score'] = float(similarities[idx])
                metadata_dict['dataset_index'] = int(idx)

                # Get text content
                text_array = ds.text[int(idx)].numpy()
                text_str = text_array.item() if text_array.size == 1 else text_array[0]

                if not text_str:
                    print("Empty text content, skipping document")
                    continue

                # Create document
                doc = Document(
                    page_content=str(text_str),
                    metadata=metadata_dict
                )
                processed_results.append(doc)

                # Debug final document
                print("\nFinal document metadata:")
                print(json.dumps(doc.metadata, indent=2))

            except Exception as e:
                print(f"Error processing result at index {idx}: {e}")
                continue

        print(f"\nTotal results processed: {len(processed_results)}")
        return processed_results

    def get_vector_store(self, vstore_name, bedrock_client):
        if vstore_name=='vectorstore_all_250_instruct':
            vstor_dir = os.path.join(self.project_root, 'data', 'vector_stores', 'vectorstore_all_250_instruct')
            embeddor = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
            vectorstore = DeepLake(dataset_path=vstor_dir, embedding_function=embeddor, read_only=False)
            return vectorstore, embeddor
        elif vstore_name=='vectorstore_all_1000_instruct':
            vstor_dir = os.path.join(self.project_root, 'data', 'vector_stores', 'vectorstore_all_1000_instruct')
            embeddor = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
            vectorstore = DeepLake(dataset_path=vstor_dir, embedding_function=embeddor, read_only=False)
            return vectorstore, embeddor
        elif vstore_name=='vectorstore_sample_250_instruct':
            vstor_dir = os.path.join(self.project_root, 'data', 'vector_stores', 'vectorstore_sample_250_instruct')
            embeddor = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
            vectorstore = DeepLake(dataset_path=vstor_dir, embedding_function=embeddor, read_only=False)
            return vectorstore, embeddor
        elif vstore_name=='vectorstore_sample_1000_instruct':
            vstor_dir = os.path.join(self.project_root, 'data', 'vector_stores', 'vectorstore_sample_1000_instruct')
            embeddor = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
            vectorstore = DeepLake(dataset_path=vstor_dir, embedding_function=embeddor, read_only=False)
            return vectorstore, embeddor
        elif vstore_name=='vectorstore_all_250_titan':
            vstor_dir = os.path.join(self.project_root, 'data', 'vector_stores', 'vectorstore_all_250_titan')
            embeddor = lambda texts: self.get_embedding_vectors(texts, bedrock_client)
            vectorstore = DeepLake(dataset_path=vstor_dir, embedding_function=embeddor, read_only=False)
            return vectorstore, embeddor

    def document_retrieval(self, embeddor, query, vectorstore, top_k):
        # Perform the search
        results = self.search_vector_store(embeddor=embeddor, query=query, vectorstore=vectorstore, top_k=top_k)
        if not results:
            print(f"\nNo results found for the query.\n")
            return query, [], 0, None, None, None, [], []
        
        # Assuming the first result is the most relevant
        num_matches = len(results)

        best_match = results[0]
        best_match_content = best_match.page_content
        best_match_filename = best_match.metadata.get('original_filename', 'Unknown')
        best_match_chunkid = best_match.metadata.get('chunk_id', -1)  # Assuming chunk IDs are stored in metadata
        best_match_score = best_match.metadata.get('similarity_score', 'Unknown')

        # Get overall data
        matches_info = []
        for match in results:
            match_content = match.page_content
            match_filename = match.metadata.get('original_filename', 'Unknown')
            match_chunkid = match.metadata.get('chunk_id', -1)
            match_score = match.metadata.get('similarity_score', 'Unknown')
            
            # Collect relevant information for each match
            matches_info.append({
                'content': match_content,
                'filename': match_filename,
                'chunk_id': match_chunkid,
                'score': match_score
            })

        all_match_filenames = list({match['filename'] for match in matches_info}) # removes duplicates
        all_match_chunkids = list({match['chunk_id'] for match in matches_info})
        all_match_scores = list({match['score'] for match in matches_info})
    
        return query, results, num_matches, best_match_content, best_match_filename, best_match_chunkid, best_match_score, all_match_filenames, all_match_chunkids, all_match_scores

    def rerank_with_custom_tfidf(self, raw_documents, query, top_k):
        def extract_documents(docs):
            documents = []
            for doc in docs:
                # Assume each doc is an object with 'metadata' and 'page_content' attributes
                metadata = doc.metadata
                content = doc.page_content
                documents.append({'metadata': metadata, 'text': content})
            return documents
        
        extracted_docs = extract_documents(raw_documents)

        def calculate_tfidf_scores(docs, query):
            documents = [doc['text'] for doc in docs]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents + [query])
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            return cosine_sim
        
        def calculate_freshness_scores(docs):
            max_date = datetime.now()
            freshness_scores = []
            for doc in docs:
                #raw_timestamp = datetime.strptime(doc['metadata'].get('timestamp', '2023-01-01'), '%Y-%m-%dT%H:%M:%S.%fZ')
                raw_timestamp = doc['metadata'].get('timestamp', '2023-01-01')
                try:
                    timestamp = datetime.strptime(raw_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
                except ValueError:
                    try:
                        timestamp = datetime.strptime(raw_timestamp, '%Y-%m-%dT%H:%M:%S')
                    except ValueError:
                        timestamp = datetime.strptime(raw_timestamp, '%Y-%m-%d')
                        
                time_diff = max_date - timestamp
                freshness_score = max(0, 1 - (time_diff.days / 365))  # Weight newer documents higher
                freshness_scores.append(freshness_score)
            return freshness_scores
        
        tfidf_scores = calculate_tfidf_scores(extracted_docs, query)
        freshness_scores = calculate_freshness_scores(extracted_docs)

        scores = []
        for idx, doc in enumerate(extracted_docs):
            keyword_coverage = sum(1 for word in set(re.findall(r'\w+', query.lower())) if word in doc['text'].lower()) / len(set(re.findall(r'\w+', query.lower())))
            total_score = (tfidf_scores[idx] * 0.5) + (freshness_scores[idx] * 0.3) + (keyword_coverage * 0.2)
            scores.append((doc, total_score))
        
        # Sort documents by total score in descending order and return top 3
        sorted_docs = sorted(scores, key=lambda x: x[1], reverse=True)
        rr_results = [doc[0] for doc in sorted_docs[:top_k]]

        # Extract 'original_filename' from the rr_results
        rr_filenames = [result['metadata']['original_filename'] for result in rr_results if 'original_filename' in result['metadata']]
        rr_best_filename = sorted_docs[0][0]['metadata']['original_filename'] if sorted_docs else None
        return rr_results, rr_filenames, rr_best_filename

    def runRetriever(self, query, top_k, vstore_name):
        # set client for amzn LLM
        config_dir = os.path.join(self.project_root, 'config')
        config = self.load_configuration(config_dir)
        bedrock_client = self.create_bedrock_client(config)

        # set vectorstore (this one uses instructor-xl)
        vectorstore, embeddor = self.get_vector_store(vstore_name, bedrock_client)

        # generate hypothetical questions and responses using claude sonnet
        hydes = self.hyde_generator(bedrock_client, query)
        
        # get questions and answers
        query1 = self.clean_hyde_docs(hydes['Original Query']) # original question
        response1 = self.clean_hyde_docs(hydes['Response1'], response=True)
        query2 = self.clean_hyde_docs(hydes['Query2'])
        response2 = self.clean_hyde_docs(hydes['Response2'], response=True)
        query3 = self.clean_hyde_docs(hydes['Query3'])
        response3 = self.clean_hyde_docs(hydes['Response3'], response=True)
            
        # error handling bad responses; e.g., lack of response due to POSSIBLE copyright concerns by the LLM
        if response3=="retract":
            if response2=="retract":
                if response1=="retract":
                    response1 = "Bad response. Please try rephrasing your query."
                    print(response1)
                    return response1
                else:
                    response2 = response1
                    response3 = response2
            else:
                response3 = response2
        
        if response2=="retract":
            if response3=="retract":
                if response1=="retract":
                    response1 = "Bad response. Please try rephrasing your query."
                    print(response1)
                    return response1
                else:
                    response2 = response1
                    response3 = response2
            else:
                response2 = response3
        
        if response1=='retract':
            if response2=='retract':
                if response3=='retract':
                    response3 = "Bad response. Please try rephrasing your query."
                    print(response3)
                    return response3
                else:
                    response2 = response3
                    response1 = response2
            else:
                response1 = response2
        
        # combine hyde doc with query
        combined_input1 = f'{query1} {response1}'
        combined_input2 = f'{query2} {response2}'
        combined_input3 = f'{query3} {response3}'

        # query vectorstore; all_match_filenamesX are uniques
        query1, results1, num_matches1, best_match_content1, best_match_filename1, best_match_chunkid1, best_match_score1, all_match_filenames1, all_match_chunkids1, all_match_scores1 = self.document_retrieval(embeddor, combined_input1, vectorstore, top_k)
        query2, results2, num_matches2, best_match_content2, best_match_filename2, best_match_chunkid2, best_match_score2, all_match_filenames2, all_match_chunkids2, all_match_scores2 = self.document_retrieval(embeddor, combined_input2, vectorstore, top_k)
        query3, results3, num_matches3, best_match_content3, best_match_filename3, best_match_chunkid3, best_match_score3, all_match_filenames3, all_match_chunkids3, all_match_scores3 = self.document_retrieval(embeddor, combined_input3, vectorstore, top_k)
        
        # reranking to get top_k
        combined_results = results1 + results2 + results3
        rr_results, rr_filenames, rr_best_filename = self.rerank_with_custom_tfidf(combined_results, query, top_k)

        return query, bedrock_client, rr_results, rr_filenames, rr_best_filename

