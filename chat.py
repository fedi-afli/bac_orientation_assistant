from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# -----------------------------
# Load embeddings (must match indexing)
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Load existing Chroma DB
# -----------------------------
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# -----------------------------
# Load LLM
# -----------------------------
llm = OllamaLLM(model="llama3")


# -----------------------------
# Retrieval + generation
# -----------------------------
def ask(question):
    # Retrieve with scores to filter low-relevance results
    # Chroma uses L2 distance: lower score = more similar
    results_with_scores = db.similarity_search_with_score(question, k=6)

    # Filter out chunks with high L2 distance (poor semantic match)
    filtered = [(doc, score) for doc, score in results_with_scores if score < 1.35]

    if not filtered:
        # Fallback: use top 3 even if scores are poor
        filtered = results_with_scores[:3]

    docs = [doc for doc, _ in filtered]
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are an expert advisor for the Tunisian University Orientation system (orientation.tn) for 2025.
You assist French Baccalaureate graduates (General and STMG tracks) from homologated Tunisian lycées.
Answer ONLY using the context below. If the information is not in the context, say so clearly.
When referencing programs, include the program code and T score formula when available.
Explain acronyms on first use (e.g., FG = Formule Globale, MF = Moyenne Finale, STMG = Bac Technologique, T = Score d'Orientation).
Do not mention that your are tied to a specific context in your response 

Context:
{context}

Question:
{question}

Answer clearly and concisely."""
    return llm.invoke(prompt)


# -----------------------------
# CLI CHAT LOOP
# -----------------------------
if __name__ == "__main__":
    print("RAG Chat ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        answer = ask(q)
        print("\nBot:", answer)
        print("-" * 50)
