import os
import pdftxt_converter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import re

# -----------------------------
# Preprocess extracted text
# -----------------------------
def preprocess_tables(text):
    # Fix common OCR whitespace problems
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove obvious noise
    noise_patterns = [
        r"<<<",
        r"Table des\s+mati[eè]res",
        r"Cliquer pour plus d'informations",
        r"^\s*\d+\s*$",      # page numbers
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text


# -----------------------------
# 1. Load TXT files and wrap them as Documents with metadata
# -----------------------------
def load_all_txt_as_documents(folder_path):
    documents = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                content = preprocess_tables(content)

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file,
                        "path": path
                    }
                )
                documents.append(doc)

    return documents


# -----------------------------
# 2. Split documents into chunks while preserving metadata
# -----------------------------
def chunk_documents(documents):
    # Larger chunk size keeps program entries (code + formula + conditions) together.
    # Higher overlap ensures formulas and their context are not split across chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=400,
        length_function=len,
        separators=[
            "\n# ",     # top-level section headers
            "\n## ",    # subsection headers
            "\n- ",     # bullet list items
            "\n* ",     # star list items
            "\n\n",     # blank lines
            "\n",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    # Raise minimum threshold to avoid orphaned micro-chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) > 200]
    return chunks


# -----------------------------
# Save chunks for inspection
# -----------------------------
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


# -----------------------------
# 3. Build Chroma DB
# -----------------------------
def build_chroma(chunks, persist_dir="chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Embedding {len(chunks)} chunks into Chroma...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return db


# -----------------------------
# 4. MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    folder_path = "refined_data"
    chroma_path = "chroma_db"

    os.makedirs(folder_path, exist_ok=True)

    if not os.listdir(folder_path):
        print("Folder is empty! Running LLMWhisperer ingestion pipeline...")
        pdftxt_converter.ingestion_pipeline()

        if not os.listdir(folder_path):
            print(f"Error: Please ensure your pdftxt_converter saves the text file into '{folder_path}'")
            exit(1)
    else:
        print("Source data ready.")

    print("Loading documents...")
    docs = load_all_txt_as_documents(folder_path)

    print("Chunking text documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    save_chunks_to_txt(chunks)
    print("Chunk preview saved to chunks_preview.txt")

    print("Building Chroma vector database...")
    db = build_chroma(chunks, chroma_path)

    print("\nRAG indexing complete! Your vector database is ready for queries.")
