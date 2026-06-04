from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = OllamaLLM(model="llama3")


def ask(question):
    results_with_scores = db.similarity_search_with_score(question, k=6)
    filtered = [(doc, score) for doc, score in results_with_scores if score < 1.35]

    if not filtered:
        filtered = results_with_scores[:3]

    # Print selected chunks for inspection
    print("\n" + "=" * 60)
    print(f"RETRIEVED {len(filtered)} CHUNKS:")
    print("=" * 60)
    for i, (doc, score) in enumerate(filtered, 1):
        print(f"\n[Chunk {i} | Score: {score:.4f} | Source: {doc.metadata.get('source', '?')}]")
        print("-" * 40)
        print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
    print("=" * 60 + "\n")

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


if __name__ == "__main__":
    print("RAG Chat ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer = ask(q)
        print("\nBot:", answer)
        print("-" * 50)