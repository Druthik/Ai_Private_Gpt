# Ai_Private_Gpt

Overview:

    This Streamlit application provides functionalities for uploading various types of documents (PDF, DOCX, TXT, HTML, MP3, WAV, OGG, MPEG), processing URLs, and answering questions based on a trained model. The project leverages libraries such as llama_index, chromadb, langchain, transformers, torch, fitz, pydub, IPython, beautifulsoup4, etc.

Features:

    Upload App:
        Users can upload different types of files.
        URLs can be processed for document retrieval.
        Functions include convert_and_save_text, process_documents, etc.

    Chat App:
        Users can ask questions.
        Trained model provides answers related to document retrieval.
        Utilizes the doc_model function to initialize a vector store index.

    Doc Model:
        Uses llama_index to create a vector store index for document retrieval.
        Embeds documents using Hugging Face models.
        Stores embeddings in a ChromaDB collection.
        Builds a VectorStoreIndex from the documents.

Usage:
    Clone the Repository:
    
    git clone https://github.com/your-username/your-repository.git
    cd your-repository

Install Dependencies:

    pip install -r requirements.txt
    
Run the Application:

    streamlit run your_script_name.py

    Replace your_script_name.py with the actual name of your Python script.

Usage Modes:

    The application offers two modes: "Upload" and "Chat."
    "Upload" mode allows users to upload files and process URLs.
    "Chat" mode enables users to ask questions and receive answers.

File Structure:

    your_script_name.py: Main script containing the Streamlit application.
    requirements.txt: List of dependencies.

Dependencies:

    List of Python libraries used in the project.

Configuration:

    Python 3.x
    Hugging Face Models:
        If you're using specific Hugging Face models, add them to the requirements. For example, if you're using a model from the         transformers library like "BAAI/bge-base-en-v1.5," you might need to include it explicitly.
    Audio Processing:
        The pydub library is used for audio processing. If you're working with specific audio file formats, you may need to               install additional libraries like pydub[ffmpeg] to handle various audio file formats.
    Streamlit:
        Ensure that the Streamlit version specified in the requirements matches the version compatible with your script.
    ChromaDB:
        The chromadb library seems to be custom or specific to your project. Ensure that it's available and correctly installed.
    torch:
        Ensure the specified version of torch is compatible with your system and other dependencies.
    Standard Libraries:
        Libraries such as os, subprocess, re, pickle, shutil, numpy, fitz, and others are part of the standard library and don't          need to be listed in the requirements.txt.

Important Notes:

    Ensure that the required resources (models, data) are available.
    Check file paths and configurations for proper setup.
    
Issues and Feedback:

    If you encounter any issues or have feedback, please open an issue.
