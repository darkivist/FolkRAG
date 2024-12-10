# Project Overview
This project is a Retrieval-Augmented Generation (RAG) system designed for research assistance and data retrieval. It integrates advanced retriever and generator models, supports vector store management, and provides an interactive interface for users. Below is an overview of the key components, features, and directories in the project.


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/darkivist/FolkRAG.git
   ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up AWS Bedrock credentials in the `config/config.ini` file:
    ```bash
    [BedRock_LLM_API]
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_SESSION_TOKEN=your_session_token
    ```
4. Set up AWS Bedrock credentions in the `config/.env` file:
    ```bash
    CONFIG_FILE=config.ini
    ```
5. Preconfigured vector stores should be available with the following names and stores in the `../data/vector_stores` directory. Vector stores not available in the repository will not function in the application unless they are constructed and housed locally or in a hosted environment.
    - `vectorstore_all_250_instruct`
    - `vectorstore_all_1000_instruct`
    - `vectorstore_all_250_titan`
    - `vectorstore_sample_250_instruct` (available in the repo)
    - `vectorstore_sample_1000_instruct` (available in the repo)

## Key Components

#### 1. Application Files
- **`generator.py`**: Manages text generation processes using AWS Bedrock's Claude.
- **`retriever.py`**: Handles data retrieval using vector stores, embedding models (`instructor-xl`, `amazon/titan`), hypothetical document generation, and reranking techniques.
- **`app.py`**: Contains the main logic for initializing and running the application.

#### 2. Vector Stores
Vector stores are utilized for efficient data retrieval and are stored in the `/data/vector_stores` directory. 
Two smaller vector stores for testing are included in the repository. Larger vector stores can be processed on 
a virtual or local machine due to size constraints.

#### 3. Streamlit Application
- **Interactive UI**: The application provides a user-friendly interface built with Streamlit.
- **Features**: 
  - Query processing and document retrieval.
  - Generative responses based on retrieved data.
  - Parameter customization for top-k documents and vector store selection.

To launch the application, use:
```bash
streamlit run src/main.py
```

## Features

#### Retriever Details
- **Vector Stores**: Supports multiple vector store configurations using embedding models.
- **HyDE Generator**: Generates hypothetical documents for enhanced query matching.
- **TF-IDF-based Reranking**: Combines relevance, freshness, and keyword coverage for optimal document selection.

#### Generator Details
- **Prompt Formulation**: Combines query context and metadata for precise responses.
- **Metadata Integration**: Dynamically incorporates metadata into prompts to enhance specificity.
- **AWS Bedrock Integration**: Leverages Claude models for text generation.



## Customization

To customize the directory structure, update the paths in `src/main.py`:
```
data_dir = os.path.join(project_root, 'data')
demo_dir = os.path.join(project_root, 'demo')
vstore_dir = os.path.join(demo_dir, 'vector_stores')
config_dir = os.path.join(project_root, 'config')
src_dir = os.path.join(project_root, 'src')
components_dir = os.path.join(src_dir, 'components')
```

For adding new components, place files in `src/components/` and ensure proper imports.


## License

This project is licensed under the MIT License. See the LICENSE file in the root directory for details.
