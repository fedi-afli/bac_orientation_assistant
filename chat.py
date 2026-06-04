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
    docs = db.similarity_search(question, k=3)


    context = "\n\n".join([d.page_content for d in docs])
    print(context)

    prompt = f"""
    You are a helpful assistant.
    Answer ONLY using the context below.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer clearly and concisely.
    """

   # return llm.invoke(prompt)


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