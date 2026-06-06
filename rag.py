import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def load_all_txt_as_documents(folder_path):
    documents = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 200]
    return chunks


def save_chunks_to_txt(chunks, output_file="chunks_preview.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK #{i}\n")
            f.write(f"Source: {chunk.metadata.get('source', 'Unknown')}\n")
            f.write(f"Path: {chunk.metadata.get('path', 'Unknown')}\n")
            f.write(f"Length: {len(chunk.page_content)} characters\n")
            f.write("-" * 80 + "\n")
            f.write(chunk.page_content)
            f.write("\n\n")


def build_chroma(chunks, persist_dir="chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Embedding {len(chunks)} chunks into Chroma...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return db


if __name__ == "__main__":

    folder_path = "refined_data"
    chroma_path = "chroma_db"

    os.makedirs(folder_path, exist_ok=True)

    if not os.listdir(folder_path):
        print(f"Error: Please place your .txt file inside the '{folder_path}' directory first.")
        exit(1)
    else:
        print("Source data ready.")

    print("Loading text documents...")
    docs = load_all_txt_as_documents(folder_path)

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    save_chunks_to_txt(chunks)
    print("Chunk preview saved to chunks_preview.txt")

    print("Building Chroma vector database...")
    db = build_chroma(chunks, chroma_path)

    print("\nRAG indexing complete! Your vector database is ready for queries.")