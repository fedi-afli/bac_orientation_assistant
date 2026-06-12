import os
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
NVIDIA_LLM_MODEL = "meta/llama-3.3-70b-instruct"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

embeddings = NVIDIAEmbeddings(
    model=NVIDIA_EMBED_MODEL,
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="END"
)

reranker = NVIDIARerank(
    model=NVIDIA_RERANK_MODEL,
    api_key=os.getenv("NVIDIA_API_KEY"),
    top_n=4
)

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

def fetch_next_sequential_chunk(db, current_doc):
    """
    Looks up the chunk that sequentially follows the current document 
    using the source file name and the chunk index.
    """
    source = current_doc.metadata.get("source")
    current_idx = current_doc.metadata.get("chunk_index")
    
    # If your ingestion pipeline didn't store indexes, fallback to standard behavior
    if source is None or current_idx is None:
        return None
        
    next_idx = int(current_idx) + 1
    
    # Query Chroma directly for the document with the next index from the same source file
    try:
        response = db.get(
            where={
                "$and": [
                    {"source": {"$eq": source}},
                    {"chunk_index": {"$eq": next_idx}}
                ]
            }
        )
        
        if response and response["documents"]:
            # Returns the text content of the next sequential chunk
            return response["documents"][0]
    except Exception:
        pass # Handle cases where it's the absolute last chunk of the document
        
    return None

def ask(question):
    # Step 1: Cast a wider net
    initial_results = db.max_marginal_relevance_search(
        question,
        k=15,          
        fetch_k=40,   
        lambda_mult=0.6  
    )

    # Step 2: Rerank down to the top 4 most semantically relevant entry points
    reranked_results = reranker.compress_documents(
        query=question,
        documents=initial_results
    )

    print("\n" + "=" * 60)
    print(f"PROCESSING AND EXPANDING {len(reranked_results)} BASE CHUNKS:")
    print("=" * 60)
    
    final_context_blocks = []
    
    for i, doc in enumerate(reranked_results, 1):
        title = doc.metadata.get('h1_original', doc.metadata.get('title', 'Sans Titre'))
        print(f"\n[Chunk {i} Base | {title}]")
        print("-" * 40)
        
        # Start with the core text found by the reranker
        full_chunk_text = doc.page_content
        
        # Check if this chunk needs its sequential successor appended
        next_chunk_text = fetch_next_sequential_chunk(db, doc)
        if next_chunk_text:
            print("⚡ Incompleteness Safety Net: Appended next sequential chunk automatically.")
            full_chunk_text += "\n\n[CONTINUATION DE LA SECTION PRÉCÉDENTE]:\n" + next_chunk_text
            
        final_context_blocks.append(full_chunk_text)
        
    print("=" * 60 + "\n")

    context = "\n\n---\n\n".join(final_context_blocks)

    prompt = f"""You are an expert advisor for the Tunisian University Orientation system (orientation.tn) for 2025.
You assist French Baccalaureate graduates (General and STMG tracks) from homologated Tunisian lycées.
Answer ONLY using the context below. If the information is not in the context, say so clearly.
When referencing programs, include the program code and T score formula when available.
Explain acronyms on first use (e.g., FG = Formule Globale, MF = Moyenne Finale, STMG = Bac Technologique, T = Score d'Orientation).
Do not mention that you are tied to a specific context in your response.
If the answer cannot be found in the context, respond ONLY with: "Cette information n'est pas disponible dans le guide."
Do NOT use any prior knowledge. Do NOT guess.

Context:
{context}

Question:
{question}

Answer clearly and concisely."""

    response = client.chat.completions.create(
        model=NVIDIA_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ NVIDIA_API_KEY missing. Check your .env file.")
        exit(1)

    print("RAG Chat ready with Context Window Expansion. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer = ask(q)
        print("\nBot:", answer)
        print("-" * 50)