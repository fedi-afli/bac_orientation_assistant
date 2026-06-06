import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from sectionning import  md_to_meta_chunks




def load_all_md_as_documents(folder_path):
    documents = []

    # Loops through the directory and filters for .md files
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".md"):
            path = os.path.join(folder_path, file)
            print(path)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file,
                        "path": path
                    }
                )
                documents.append(doc)

    return documents

def chunk_documents(documents):
    all_chunks=[]
    for doc in documents:
        all_chunks.extend(md_to_meta_chunks(doc))
    return all_chunks





def build_chroma(all_chunks, persist_dir="chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Embedding {len(chunks)} chunks into Chroma...")
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return db


if __name__ == "__main__":

    folder_path = "docs"
    chroma_path = "chroma_db"

    os.makedirs(folder_path, exist_ok=True)

    if not os.listdir(folder_path):
        print(f"Error: Please place your .txt file inside the '{folder_path}' directory first.")
        exit(1)
    else:
        print("Source data ready.")

    print("Loading text documents...")
    docs = load_all_md_as_documents(folder_path)

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} text chunks.")



    print("Building Chroma vector database...")
    db = build_chroma(chunks, chroma_path)

    print("\nRAG indexing complete! Your vector database is ready for queries.")