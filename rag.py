import os
# Remove pdftxt_converter if it's no longer needed for pre-processed markdown files
# import pdftxt_converter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# 1. Load MD files and wrap them as Documents with metadata
# -----------------------------
def load_all_md_as_documents(folder_path):
    documents = []

    for file in sorted(os.listdir(folder_path)):
        # CHANGE: Look for .md files instead of .txt
        if file.endswith(".md"):
            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

                # NOTE: Bypassed preprocess_tables because the generated .md
                # file is already structured and doesn't contain raw OCR noise.

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
# 2. Split documents using native Markdown Splitter
# -----------------------------
def chunk_documents(documents):
    # CHANGE: Switched to LangChain's native Markdown splitter rules
    # to perfectly handle headers, lists, and markdown tables.
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,  # Lowered slightly for more granular text chunks
        chunk_overlap=300
    )

    chunks = splitter.split_documents(documents)

    # Filter out empty or micro-chunks if necessary
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

    # Make sure you drop 'guide_orientation_2025_rag_optimized.md' inside 'refined_data/'
    if not os.listdir(folder_path):
        print(f"Error: Please place your .md file inside the '{folder_path}' directory first.")
        exit(1)
    else:
        print("Source data ready.")

    print("Loading Markdown documents...")
    # CHANGE: Call the MD loader
    docs = load_all_md_as_documents(folder_path)

    print("Chunking Markdown documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    save_chunks_to_txt(chunks)
    print("Chunk preview saved to chunks_preview.txt")

    print("Building Chroma vector database...")
    db = build_chroma(chunks, chroma_path)

    print("\nRAG indexing complete! Your vector database is ready for queries.")