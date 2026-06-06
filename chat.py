from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = OllamaLLM(model="llama3")
""" 
The fix is Maximum Marginal Relevance (MMR) search instead of plain similarity search. 
MMR balances relevance and diversity — it penalizes chunks that are too similar to ones already selected.
 """
def ask(question):
    results = db.max_marginal_relevance_search(
        question,
        k=4,
        fetch_k=20,
        lambda_mult=0.7,
    )

    context = "\n\n".join([d.page_content for d in results])
    print(context)

    prompt = f"""You are an expert advisor for the Tunisian University Orientation system (orientation.tn) for 2025.
You assist French Baccalaureate graduates (General and STMG tracks) from homologated Tunisian lycées.
Answer ONLY using the context below. If the information is not in the context, say so clearly.
When referencing programs, include the program code and T score formula when available.
Explain acronyms on first use (e.g., FG = Formule Globale, MF = Moyenne Finale, STMG = Bac Technologique, T = Score d'Orientation).
Do not mention that you are tied to a specific context in your response.
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