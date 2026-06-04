import os
# Make sure your LLMWhisperer extraction script is named pdftxt_converter.py
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

    # Split into lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    processed = []
    current_domain = ""

    for line in lines:

        # Detect domain headers
        if any(keyword in line for keyword in [
            "Lettres",
            "Sciences Exactes",
            "Architecture",
            "Sciences Juridiques",
            "Sciences Economiques",
            "Sciences de la Santé",
            "Sciences Agronomiques"
        ]):
            current_domain = line
            processed.append(f"\n=== Domaine: {current_domain} ===\n")
            continue

        # Detect probable formation rows
        m = re.search(
            r"(\d{5})\s+(.*?)\s+(Bac G[ée]n[ée]ral|STMG)",
            line
        )

        if m:
            code = m.group(1)
            before_bac = m.group(2)
            serie = m.group(3)

            processed.append(
                f"Code: {code}\n"
                f"Description: {before_bac}\n"
                f"Série: {serie}\n"
            )
        else:
            processed.append(line)

    return "\n".join(processed)
# -----------------------------
# 1. Load TXT files and wrap them as Documents with metadata
# -----------------------------
def load_all_txt_as_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

                # Preprocess before chunking
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
    # For highly dense, tabular/formula documents, a slightly smaller chunk size
    # combined with a larger overlap helps keep table rows and context together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        separators=["\f", "\n\n", "\n", " ", ""]  # \f handles form-feed/page breaks if present
    )

    # split_documents automatically passes the metadata down to each individual chunk
    chunks = splitter.split_documents(documents)
    chunks = [
        c for c in chunks
        if len(c.page_content.strip()) > 80
    ]
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
    # Ensure Ollama is running locally with the nomic model pulled
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Embedding {len(chunks)} chunks into Chroma...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Note: db.persist() is deprecated in newer LangChain versions
    # as Chroma persists data automatically upon creation.
    return db


# -----------------------------
# 4. MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    folder_path = "refined_data"
    chroma_path = "chroma_db"

    # Step 1: Ensure directory exists and check if data is there
    os.makedirs(folder_path, exist_ok=True)

    if not os.listdir(folder_path):
        print("Folder is empty! Running LLMWhisperer ingestion pipeline...")
        pdftxt_converter.ingestion_pipeline()

        # Double check if the pipeline actually created the file in the right directory
        if not os.listdir(folder_path):
            print(f"Error: Please ensure your pdftxt_converter saves the text file into '{folder_path}'")
            exit(1)
    else:
        print("Source data ready.")

    # Step 2: Load text files as Documents
    print("Loading documents...")
    docs = load_all_txt_as_documents(folder_path)

    # Step 3: Chunking
    print("Chunking text documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    save_chunks_to_txt(chunks)
    print("Chunk preview saved to chunks_preview.txt")

    # Step 4: Build Chroma DB
    print("Building Chroma vector database...")
    db = build_chroma(chunks, chroma_path)

    print("\nRAG indexing complete! 🚀 Your vector database is ready for queries.")