import streamlit as st
import os
import subprocess
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from glob import glob
import pickle
import shutil
import numpy as np
from llama_index.storage.storage_context import StorageContext
import fitz
from IPython.display import Markdown,display
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from llama_index.response.schema import Response
from docx import Document
from bs4 import BeautifulSoup
from pydub import AudioSegment

def links_model(link):
    output_directory = './docs_generated/'
    os.makedirs(output_directory, exist_ok=True)
    #url = link[0]
    filename = os.path.join(output_directory, f"{link.split('/')[-1]}.txt")
    # print("Generating document for url")
    loaders = UnstructuredURLLoader(urls=[link])
    data = loaders.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    # Open the file in write mode ('w') with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as file:
        # Iterate through each document in the list
        for document in docs:
            # Assuming the Document class has a get_text() method
            document_text = document.copy()
            # Write the document's text to the file
            file.write(str(document_text) + '\n')
	return None
			

def pdfs_model(pdf_url):
    output_directory = './docs_generated/'
    os.makedirs(output_directory, exist_ok=True)
    # File path for the downloaded PDF
    pdf_filepath = os.path.join(output_directory, f"{pdf_url.split('/')[-1]}.pdf")
    # Check if the file already exists
    if not os.path.exists(pdf_filepath):
        # Use subprocess to run the curl command
        subprocess.run(["curl", pdf_url, "--output", pdf_filepath])
    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_filepath)
    # Extract text from all pages
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text()
    # Save extracted text to file
    output_filepath = os.path.join(output_directory, f"{pdf_url.split('/')[-1]}.txt")
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(extracted_text)
    doc.close()
    return None
	
def process_documents(URL):
    output_directory = './docs_generated/'
    os.makedirs(output_directory, exist_ok=True)

    #for item in data_array:
        #url = item[0]
    is_pdf = URL.endswith('.pdf')

    if is_pdf:
        # If the URL ends with '.pdf', treat it as a PDF URL
        pdfs_model(url, output_directory)
    else:
        # Otherwise, treat it as a regular URL
        links_model([(url,)], output_directory)
			

def convert_and_save_text(file):
    # File path for the uploaded file
    file_filepath = os.path.join('./uploads', file.name)

    # Save the uploaded file
    with open(file_filepath, 'wb') as f:
        f.write(file.getvalue())

    # Perform conversion based on file type
    file_extension = file.name.split('.')[-1].lower()

    if file_extension == 'pdf':
        # PDF: Use PyMuPDF for text extraction
        doc = fitz.open(file_filepath)
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text()
        doc.close()
		output_directory = './docs_generated/'
		os.makedirs(output_directory, exist_ok=True)
		output_filepath = os.path.join(output_directory, f"{file.name.split('.')[0]}_converted.txt")
		with open(output_filepath, 'w', encoding='utf-8') as f:
			f.write(extracted_text)
			
    elif file_extension == 'docx':
        # DOCX: Use python-docx for text extraction
        doc = Document(file_filepath)
        extracted_text = " ".join([paragraph.text for paragraph in doc.paragraphs])
		output_directory = './docs_generated/'
		os.makedirs(output_directory, exist_ok=True)
		output_filepath = os.path.join(output_directory, f"{file.name.split('.')[0]}_converted.txt")
		with open(output_filepath, 'w', encoding='utf-8') as f:
			f.write(extracted_text)
			
    elif file_extension == 'html':
        # HTML: Use BeautifulSoup for text extraction
        with open(file_filepath, 'r', encoding='utf-8') as html_file:
            soup = BeautifulSoup(html_file, 'html.parser')
            extracted_text = soup.get_text()
		output_directory = './docs_generated/'
		os.makedirs(output_directory, exist_ok=True)
		output_filepath = os.path.join(output_directory, f"{file.name.split('.')[0]}_converted.txt")
		with open(output_filepath, 'w', encoding='utf-8') as f:
			f.write(extracted_text)

    elif file_extension in ['txt', 'text']:
        # Plain Text: Read the content directly
        with open(file_filepath, 'r', encoding='utf-8') as text_file:
            extracted_text = text_file.read()
		    # Save extracted text to file
		output_directory = './docs_generated/'
		os.makedirs(output_directory, exist_ok=True)
		output_filepath = os.path.join(output_directory, f"{file.name.split('.')[0]}_converted.txt")
		with open(output_filepath, 'w', encoding='utf-8') as f:
			f.write(extracted_text)

    elif file_extension in ['mp3', 'wav', 'ogg', 'mpeg']:
        # Audio files: Use pydub for text extraction (assuming automatic speech recognition)
        audio = audio_text(file_filepath)

    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

    # Return the extracted text
    return None
	
def audio_text(audio_filepath):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
 
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Perform transcription
    result = pipe(audio_filepath)

    output_directory = './docs_generated/'
    os.makedirs(output_directory, exist_ok=True)
    audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
 
    # Define the output file path using the audio file name
    output_filepath = os.path.join(output_directory, f"{audio_basename}.txt")
 
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(result['text'])

    return None


def doc_model():
    input_directory = './docs_generated/'
    # proc_directory = './proc_docs/'
    # os.makedirs(proc_directory, exist_ok=True)
    # index_directory = "./index_directory"
    # os.makedirs(index_directory, exist_ok=True)
    ###############################################################################################

    os.environ["OPENAI_API_KEY"] = "Replace with your OpenAI API Key"

    ###############################################################################################
    # Use glob to get a list of all files in the directory
    input_files = glob(os.path.join(input_directory, "*"))

    # Load data from all files in the specified directory
    documents = SimpleDirectoryReader(
        input_files=input_files
    ).load_data()
	#response = requests.get("https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json", verify=False)
    # define embedding function
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # Create a persistent client and a new collection
    db = chromadb.PersistentClient(path="./chromaDB")
    collection_name = "orcass_papers"
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model,
                                                   chunk_size=400,
                                                   chunk_overlap=10)

    # Create a VectorStoreIndex instance
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )

    return index

####################################################################################
# Define a Streamlit app
def main():
    st.title("*Unstable*")

    # Navigation bar for switching between training and asking questions
    app_mode = st.sidebar.radio("Select App Mode", ["Upload", "Chat"])

    if app_mode == "Upload":
        upload_app()
    elif app_mode == "Chat":
        chat_app()



# Training app
def upload_app():
    #Upload files
    upload_file = st.file_uploader("Upload a file ", type=["pdf", "docx", "txt", "text", "mp3", "wav", "ogg", "mpeg"])

    # Add input for URLs
    links_input = st.text_input("Enter a link:")

    # Add a button to trigger processing
    if st.button("upload data"):
        # Process uploaded PDF
        #if uploaded_pdf:
            # Process the PDF using your existing code
            #pass

        # Process URL
        if links_input:
            #links = [links_input.split(",")]
            links = [[links_input]]
            process_documents(links)


        # Process uploaded audio file
        if uploaded_file:
            # Process the audio file using your existing code
			convert_and_save_text(uploaded_file)
            #pass

        # Get the question from the sidebar
        # Train the model with the uploaded data and question
        # Replace the following line with your actual training code


        # Display results
        #st.write("Question:", question)
        #st.write("Answer:", index)
    #return index

# Asking questions app
def chat_app():
    index = doc_model()
    # Get the question from the user
    question = st.text_input("Ask your question")

    # Add a button to trigger processing
    if st.button("Get Answer"):
        # Get the answer using your trained model
        # Replace the following line with your actual inference code
        query_answer = index.as_query_engine()
        response = query_answer.query(question)
        # Display the answer
        st.write("Question:", question)
        # Apply custom CSS to set the background color of text_area
        custom_css = f"""
            <style>
                div[role="textbox"][aria-multiline="true"] div {{
                    background-color: rgb(14, 17, 23);
                    color: white; /* Set text color to white for better visibility */
                }}
            </style>
            """

        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown(f'<div class="custom-text-area">{response}</div>', unsafe_allow_html=True)
        #st.text(response)
        #st.text_area("Answer:", value=response, height=1000)


# Run the Streamlit app
if __name__ == "__main__":
    main()