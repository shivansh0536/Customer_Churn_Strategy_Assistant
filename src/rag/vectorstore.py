import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore():
    persist_directory = "src/rag/chroma_db"
    
    # Initialize embeddings
    # Make sure GROQ_API_KEY is in the environment
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If the vector database already exists, load it
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        return vectorstore
        
    # If not, create it by loading the knowledge base
    print("Building Vector Store...")
    kb_path = "knowledge_base/strategies.txt"
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")
        
    loader = TextLoader(kb_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["---", "\n\n", "\n"]
    )
    docs = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print("Vector Store built and saved successfully.")
    
    return vectorstore

def retrieve_strategies(query, k=3):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    # Test retrieval
    import dotenv
    dotenv.load_dotenv()
    results = retrieve_strategies("User is not active and hasn't logged in recently.")
    print("Retrieved Strategies:")
    for r in results:
        print(r)
        print("-" * 20)
