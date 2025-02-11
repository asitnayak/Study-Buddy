from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from colorama import Fore, Style
from langchain.schema import Document
import os
import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["TAVILY_API_KEY"] = "your_key"

file_path = os.path.join('processed_files', 'combined_text.txt')

def get_retriever():
    combined_text = None
    with open(file_path, 'r', encoding='utf-8') as f:
        combined_text = f.read()

    # print(len(combined_text))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_text(combined_text)
    # print(len(doc_splits))

    documents = [
        Document(page_content=chunk, metadata={"source": 'combined_text.txt', "chunk_index": idx})
        for idx, chunk in enumerate(doc_splits)
        ]

    # Define the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

    # Define the embedding model
    embed = OpenAIEmbeddings()

    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embed)


    db = Chroma.from_documents(
        documents=documents,
        persist_directory=persistent_directory,
        embedding=embed
    )

    retriever = db.as_retriever(k=20)
    
    # return
    return retriever

# print(Fore.LIGHTYELLOW_EX + "retriever script running complete." + Style.RESET_ALL)

if __name__ == "__main__":
    # Only execute when running this script directly
    this_retriever = get_retriever()
    print(Fore.LIGHTYELLOW_EX + "store_into_db script completed running.")