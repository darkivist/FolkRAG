from typing import Dict, List, Any
from langchain_community.vectorstores import DeepLake
from langchain.schema import Document
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
import os
import shutil
import re
import gc
import torch
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings
from logging_config import setup_logging
from metadata_processor import process_metadata, clean_metadata_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from datetime import datetime

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


class AWSCredentialManager:
    def __init__(self, logger=None):
        self.logger = logger
        self._client = None

        # Ask user to check credentials at startup
        input("Please refresh your Bedrock credentials in config.ini and press Enter to continue...")

        if self.logger:
            self.logger.info("Initializing AWS credential manager")

        # Load initial configuration
        self.config = self.load_configuration(initial=True)

    def load_configuration(self, initial=False):
        """Load configuration from config files."""
        try:
            components_dir = os.path.dirname(os.path.dirname(__file__))  # This gets us to src/
            project_root = os.path.dirname(components_dir)  # This gets us to project root
            config_dir = os.path.join(project_root, 'config')
            config_file = os.path.join(config_dir, 'config.ini')

            if initial:
                print(f"Loading config from: {config_file}")

            config = ConfigParser(interpolation=ExtendedInterpolation())
            config.read(config_file)
            return config
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading configuration: {e}")
            print(f"Error loading configuration: {e}")
            raise

    def get_bedrock_client(self):
        """Get a Bedrock client with current credentials."""
        try:
            config = self.load_configuration()
            session = boto3.Session(
                aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
                aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
                aws_session_token=config['BedRock_LLM_API']['aws_session_token']
            )
            self._client = session.client("bedrock-runtime", region_name="us-east-1")
            return self._client

        except ClientError as e:
            if 'ExpiredToken' in str(e) or 'ExpiredTokenException' in str(e):
                if self.logger:
                    self.logger.info("Credentials expired, waiting for user to update them...")
                print("\nBedrock credentials expired. Please update config.ini and press Enter to continue...")
                input()
                # Retry with new credentials
                return self.get_bedrock_client()
            print(f"Failed to create Bedrock client: {str(e)}")
            raise


class RAGRetriever:
    def __init__(self, dataset_path: str = './my_deeplake', model_name: str = 'instructor', chunk_size: int = 1000,
                 logger=None):
        """Initialize RAGRetriever with tensor parameters."""
        self.logger = logger or setup_logging()
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size

        print(f"\nInitializing RAG Retriever with {model_name} model and chunk size {chunk_size}")

        # Initialize embedding model based on selection
        if model_name == 'instructor':
            self.embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        elif model_name == 'mini':
            self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        elif model_name == 'titan':
            self.credential_manager = AWSCredentialManager(logger=self.logger)
            self.embeddings = BedrockEmbeddings(
                client=self.credential_manager.get_bedrock_client(),
                region_name="us-east-1",
                model_id="amazon.titan-embed-text-v2:0")
        else:
            print('Model name not recognized. Implementing default HuggingFace Embedding model')
            self.embeddings = HuggingFaceEmbeddings()

        self.vectorstore = None
        self.documents = []

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents with enhanced metadata handling and logging."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size * 15 // 100,
            length_function=len,
        )

        timecode_pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\] '
        chunked_documents = []
        total_processed = 0

        for doc in documents:
            try:
                # Clean metadata
                clean_metadata = clean_metadata_dict(doc.metadata)

                # Process content based on file type
                is_transcript = clean_metadata.get('file_type') == 'transcript'
                if is_transcript:
                    lines = doc.page_content.split('\n')
                    cleaned_lines = [re.sub(timecode_pattern, '', line) for line in lines]
                    content = '\n'.join(line for line in cleaned_lines if line.strip())
                    print(f"Cleaned transcript content for {clean_metadata.get('original_filename', 'unknown')}")
                else:
                    content = doc.page_content

                if not content.strip():
                    print(
                        f"Warning: Empty content after cleaning for document: {clean_metadata.get('original_filename', 'unknown')}")
                    continue

                # Split content into chunks
                chunks = text_splitter.split_text(content)

                # Create chunk documents
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **clean_metadata,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }

                    chunked_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    chunked_documents.append(chunked_doc)

                total_processed += 1
                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} documents into {len(chunked_documents)} chunks")

            except Exception as e:
                print(f"Error chunking document {doc.metadata.get('original_filename', 'unknown')}: {e}")
                continue

        print(f"Final chunk count: {len(chunked_documents)} from {total_processed} documents")
        return chunked_documents

    def initialize_vectorstore(self, delete_existing: bool = False) -> None:
        try:
            if self.vectorstore is not None:
                print("Cleaning up existing vectorstore reference")
                self.vectorstore = None

            if delete_existing and os.path.exists(self.dataset_path):
                print(f"Deleting existing DeepLake dataset at {self.dataset_path}")
                shutil.rmtree(self.dataset_path)
                print("Dataset deleted successfully")

            import deeplake
            print("Creating new dataset with all tensors")

            ds = deeplake.empty(self.dataset_path, overwrite=True)

            # Core tensors
            ds.create_tensor('text', dtype='str', sample_compression=None)
            ds.create_tensor('embedding', dtype='float32', sample_compression=None)
            ds.create_tensor('metadata', dtype='str', sample_compression=None)
            ds.create_tensor('id', dtype='str', sample_compression=None)

            # Document structure
            ds.create_tensor('original_filename', dtype='str', sample_compression=None)
            ds.create_tensor('file_type', dtype='str', sample_compression=None)
            ds.create_tensor('chunk_id', dtype='int64', sample_compression=None)
            ds.create_tensor('total_chunks', dtype='int64', sample_compression=None)

            # Basic identification
            ds.create_tensor('call_number', dtype='str', sample_compression=None)
            ds.create_tensor('title', dtype='str', sample_compression=None)
            ds.create_tensor('date', dtype='str', sample_compression=None)
            ds.create_tensor('created_published', dtype='str', sample_compression=None)
            ds.create_tensor('language', dtype='str', sample_compression=None)
            ds.create_tensor('type', dtype='str', sample_compression=None)

            # Contributors and sources
            ds.create_tensor('contributors', dtype='str', sample_compression=None)
            ds.create_tensor('creator', dtype='str', sample_compression=None)
            ds.create_tensor('repository', dtype='str', sample_compression=None)
            ds.create_tensor('collection', dtype='str', sample_compression=None)
            ds.create_tensor('source_collection', dtype='str', sample_compression=None)

            # Content details
            ds.create_tensor('description', dtype='str', sample_compression=None)
            ds.create_tensor('notes', dtype='str', sample_compression=None)
            ds.create_tensor('subjects', dtype='str', sample_compression=None)
            ds.create_tensor('original_format', dtype='str', sample_compression=None)
            ds.create_tensor('online_formats', dtype='str', sample_compression=None)

            # Rights and access
            ds.create_tensor('rights', dtype='str', sample_compression=None)
            ds.create_tensor('access_restricted', dtype='str', sample_compression=None)
            ds.create_tensor('locations', dtype='str', sample_compression=None)
            ds.create_tensor('url', dtype='str', sample_compression=None)

            # Collection and catalog fields
            ds.create_tensor('collection_title', dtype='str', sample_compression=None)
            ds.create_tensor('collection_date', dtype='str', sample_compression=None)
            ds.create_tensor('collection_abstract', dtype='str', sample_compression=None)
            ds.create_tensor('series_title', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_title', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_creator', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_date', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_description', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_subjects', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_notes', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_language', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_genre', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_contributors', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_repository', dtype='str', sample_compression=None)
            ds.create_tensor('catalog_collection_id', dtype='str', sample_compression=None)

            self.vectorstore = DeepLake(
                dataset_path=self.dataset_path,
                embedding_function=self.embeddings,
                read_only=False
            )

            print("Vectorstore initialized successfully")

        except Exception as e:
            print(f"Error during vectorstore initialization: {e}")
            raise

    def preprocess_metadata(self, metadata: Dict[str, Dict]) -> Dict[str, Dict]:
        """Preprocess metadata by stripping extensions from keys."""
        return {key.split('.')[0]: value for key, value in metadata.items()}

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        """Load all documents with simplified metadata matching"""
        # Preprocess metadata once at the start
        self.metadata = self.preprocess_metadata(metadata)

        # Set up logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'metadata_matching_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        def log_print(message):
            print(message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

        def get_base_filename(filename):
            # Strip all extensions - everything after the first period
            return filename.split('.')[0]

        log_print("\n=== Metadata Matching Analysis ===")
        log_print(f"Total metadata entries: {len(self.metadata)}")
        log_print("\nSample metadata keys (first 10):")
        sample_keys = list(self.metadata.keys())[:10]
        for key in sample_keys:
            log_print(f"- {key}")

        self.documents = []
        txt_dir = os.path.join(data_dir, 'txt')
        transcripts_dir = os.path.join(data_dir, 'transcripts')
        ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')

        docs_with_metadata = 0
        docs_without_metadata = 0

        total_files = 0
        for directory in [txt_dir, transcripts_dir, ocr_dir]:
            if not os.path.exists(directory):
                continue
            total_files += len([f for f in os.listdir(directory) if f.endswith('.txt')])

        log_print(f"\nTotal files to process: {total_files}")

        for directory in [txt_dir, transcripts_dir, ocr_dir]:
            if not os.path.exists(directory):
                log_print(f"\nDirectory not found: {directory}")
                continue

            dir_type = 'transcript' if 'transcripts' in directory else 'ocr' if 'txtConversion' in directory else 'txt'
            log_print(f"\nProcessing directory: {directory}")
            log_print(f"Directory type: {dir_type}")

            for filename in os.listdir(directory):
                if not filename.endswith('.txt'):
                    continue

                file_path = os.path.join(directory, filename)
                log_print(f"\nProcessing: {filename}")
                log_print(f"Directory type: {dir_type}")

                # Get base filename without any extensions
                base_filename = get_base_filename(filename)
                log_print(f"Base filename: {base_filename}")

                # Set file type based on directory
                file_type = dir_type

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    # Check if metadata exists using base filename
                    metadata_exists = base_filename in self.metadata
                    log_print(f"Metadata match found: {metadata_exists}")

                    if metadata_exists:
                        log_print("Creating document with complete metadata")
                        metadata_entry = self.metadata[base_filename]
                        log_print("Metadata content:")
                        log_print(json.dumps(metadata_entry, indent=2))

                        doc_metadata = {
                            'original_filename': filename,
                            'base_filename': base_filename,
                            'file_type': file_type,
                            'source_directory': os.path.basename(directory),
                            'metadata_status': 'complete',
                            'title': metadata_entry.get('title', 'N/A'),
                            'date': metadata_entry.get('date', 'N/A'),
                            'contributors': metadata_entry.get('contributors', 'N/A'),
                            'creator': metadata_entry.get('creator', 'N/A'),
                            'description': metadata_entry.get('description', ''),
                            'subjects': metadata_entry.get('subjects', ''),
                            'call_number': metadata_entry.get('call_number', ''),
                            'collection': metadata_entry.get('collection', ''),
                            'repository': metadata_entry.get('repository', ''),
                            'rights': metadata_entry.get('rights', ''),
                            'language': metadata_entry.get('language', ''),
                            'type': metadata_entry.get('type', ''),
                            'original_format': metadata_entry.get('original_format', ''),
                            'online_formats': metadata_entry.get('online_formats', ''),
                            'access_restricted': metadata_entry.get('access_restricted', ''),
                            'url': metadata_entry.get('url', ''),
                            'collection_title': metadata_entry.get('collection_title', ''),
                            'collection_date': metadata_entry.get('collection_date', ''),
                            'collection_abstract': metadata_entry.get('collection_abstract', ''),
                            'source_collection': metadata_entry.get('source_collection', ''),
                            'catalog_title': metadata_entry.get('catalog_title', ''),
                            'catalog_creator': metadata_entry.get('catalog_creator', ''),
                            'catalog_date': metadata_entry.get('catalog_date', ''),
                            'catalog_description': metadata_entry.get('catalog_description', ''),
                            'catalog_subjects': metadata_entry.get('catalog_subjects', ''),
                            'catalog_contributors': metadata_entry.get('catalog_contributors', ''),
                            'catalog_notes': metadata_entry.get('catalog_notes', ''),
                            'catalog_language': metadata_entry.get('catalog_language', ''),
                            'catalog_genre': metadata_entry.get('catalog_genre', ''),
                            'catalog_repository': metadata_entry.get('catalog_repository', ''),
                            'catalog_collection_id': metadata_entry.get('catalog_collection_id', '')
                        }
                        docs_with_metadata += 1

                    else:
                        log_print("No metadata match found, checking similar keys:")
                        # Log similar keys for debugging
                        similar_keys = [k for k in self.metadata.keys() if base_filename.lower() in k.lower()]
                        if similar_keys:
                            log_print("Found similar keys:")
                            for key in similar_keys:
                                log_print(f"  - {key}")

                        doc_metadata = {
                            'original_filename': filename,
                            'base_filename': base_filename,
                            'file_type': file_type,
                            'source_directory': os.path.basename(directory),
                            'metadata_status': 'missing',
                            'title': 'N/A',
                            'date': 'N/A',
                            'contributors': 'N/A'
                        }
                        docs_without_metadata += 1

                    doc = Document(page_content=content, metadata=doc_metadata)
                    self.documents.append(doc)

                except Exception as e:
                    log_print(f"Error processing file {filename}: {e}")

        # Print summary statistics
        log_print("\n=== Loading Summary ===")
        log_print(f"Documents with metadata: {docs_with_metadata}")
        log_print(f"Documents without metadata: {docs_without_metadata}")
        log_print(f"Total documents loaded: {len(self.documents)}")
        if len(self.documents) > 0:
            metadata_rate = (docs_with_metadata / len(self.documents)) * 100
            log_print(f"Metadata match rate: {metadata_rate:.2f}%")

        log_print(f"\nDetailed debug log written to: {log_file}")
        return self.documents


    def generate_embeddings(self, documents: List[Document], batch_size: int = 1000) -> None:
        """Generate embeddings with proper metadata handling and debugging"""
        if self.vectorstore is None:
            print("Vectorstore not initialized. Call initialize_vectorstore first.")
            raise

        try:
            total_docs = len(documents)
            print(f"\nProcessing {total_docs} documents in batches of {batch_size}")
            ds = self.vectorstore.vectorstore.dataset

            # Set up logging
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'embedding_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

            def log_print(message):
                print(message)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')

            # Process documents in batches
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch = documents[batch_start:batch_end]

                try:
                    # Initialize lists for batch data
                    texts = []
                    embeddings = []
                    metadata_list = []
                    tensor_data = {field: [] for field in ds.tensors.keys() if
                                   field not in ['text', 'embedding', 'metadata', 'id']}

                    # Process documents in current batch
                    i = 0
                    while i < len(batch):
                        doc = batch[i]
                        try:
                            # Update client before embedding if needed
                            if hasattr(self, 'credential_manager'):
                                self.embeddings.client = self.credential_manager.get_bedrock_client()

                            log_print(f"Processing document {batch_start + i + 1}/{total_docs}")

                            # Debug print metadata for first doc in batch
                            if i == 0:
                                log_print("\nFirst document metadata:")
                                log_print(json.dumps(doc.metadata, indent=2))

                            # Generate embedding
                            embedding = self.embeddings.embed_query(doc.page_content)

                            # Store text and embedding
                            texts.append(doc.page_content)
                            embeddings.append(embedding)

                            # Store complete metadata as JSON
                            metadata_list.append(json.dumps(doc.metadata))

                            # Process each metadata field for individual tensors
                            for field in tensor_data:
                                value = None

                                # Handle special fields
                                if field == 'chunk_id':
                                    value = doc.metadata.get('chunk_id', 0)
                                elif field == 'total_chunks':
                                    value = doc.metadata.get('total_chunks', 0)
                                else:
                                    # Get value from metadata, with proper fallbacks
                                    value = doc.metadata.get(field, '')

                                    # Convert complex types to strings
                                    if isinstance(value, (list, dict)):
                                        value = json.dumps(value)
                                    elif value is None:
                                        value = ''

                                tensor_data[field].append(value)

                                # Debug print for first document's tensor values
                                if i == 0:
                                    log_print(f"Field {field} value: {value}")

                            i += 1

                        except Exception as e:
                            log_print(f"Error processing document {batch_start + i + 1}: {e}")
                            i += 1
                            continue

                    # Add batch to dataset
                    if texts:
                        try:
                            # Debug print before adding to dataset
                            log_print("\nTensor data about to be added:")
                            for field, values in tensor_data.items():
                                if values:
                                    log_print(f"{field} (first value): {values[0]}")
                                    log_print(f"{field} length: {len(values)}")

                            with ds:
                                # Extend core tensors
                                ds.text.extend(texts)
                                ds.embedding.extend(embeddings)
                                ds.metadata.extend(metadata_list)

                                # Extend metadata tensors
                                for field, values in tensor_data.items():
                                    log_print(f"\nExtending tensor {field}")
                                    log_print(f"Values length: {len(values)}")
                                    if values:
                                        log_print(f"Sample value: {values[0]}")
                                    tensor = getattr(ds, field)
                                    tensor.extend(values)

                            # Debug print after adding to dataset
                            log_print("\nVerifying stored data:")
                            last_idx = len(ds.text) - 1
                            for field in tensor_data.keys():
                                try:
                                    stored_value = ds[field][last_idx].numpy()
                                    log_print(f"Stored {field}: {stored_value}")
                                except Exception as e:
                                    log_print(f"Error checking {field}: {e}")

                            log_print(f"Successfully added batch of {len(texts)} documents")

                        except Exception as e:
                            log_print(f"Error adding batch to dataset: {e}")
                            raise

                except Exception as e:
                    log_print(f"Error processing batch: {str(e)}")
                    raise

            log_print(f"\nCompleted processing {total_docs} documents")
            log_print(f"Debug log written to: {log_file}")

        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise

    def process_with_checkpoints(self, documents: List[Document], batch_size: int = 1000) -> None:
        """Process documents with checkpointing for recovery from failures."""
        # First chunk the documents
        chunked_docs = self.chunk_documents(documents)
        if not chunked_docs:
            raise ValueError("No chunks generated from documents")

        # Then proceed with embedding generation
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_file = os.path.join('checkpoints',
                                       f'embedding_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        start_idx = 0
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    start_idx = checkpoint_data.get('last_processed_index', 0)
                    print(f"Resuming from checkpoint at index {start_idx}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        for batch_start in range(start_idx, len(chunked_docs), batch_size):
            batch_end = min(batch_start + batch_size, len(chunked_docs))
            batch = chunked_docs[batch_start:batch_end]

            try:
                print(f"\nProcessing batch {batch_start}-{batch_end} of {len(chunked_docs)}")
                self.generate_embeddings(batch)

                checkpoint_data = {
                    'last_processed_index': batch_end,
                    'total_documents': len(chunked_docs),
                    'timestamp': datetime.now().isoformat()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

                print(f"Checkpoint saved at document {batch_end}")

            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                checkpoint_data = {
                    'last_processed_index': batch_start,
                    'total_documents': len(chunked_docs),
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                raise

    def is_empty(self) -> bool:
        """Check if the vectorstore has any documents."""
        if self.vectorstore is None:
            return True
        return len(self.vectorstore.vectorstore.dataset) == 0
