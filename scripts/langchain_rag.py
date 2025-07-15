import os
from typing import List

# --- Key LangChain Imports ---
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # Import PromptTemplate

# -----------------------------------------------------------------------------
# 1) Build the RetrievalQA chain with a CUSTOM PROMPT
# -----------------------------------------------------------------------------
def build_chain():
    """
    Builds and returns a RetrievalQA chain with a custom prompt.
    """
    # --- Configuration ---
    CHROMA_PATH = os.path.join("data", "chroma_db")
    COLLECTION_NAME = "ai_second_brain"
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama3"

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embedding model loaded.")

    print("Loading Chroma vector store...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    print("Vector store loaded.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    llm = Ollama(model=LLM_MODEL)

    # --- THIS IS THE CRITICAL FIX ---
    # Create a custom prompt template that matches your working script
    prompt_template = """You are an academic research assistant. Use the following context excerpts to answer the question.

Context:
{context}

Question: {question}

Answer in a clear, detailed manner:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # --- END OF FIX ---

    # Build the RetrievalQA chain, passing in the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT} # Pass the custom prompt here
    )
    
    print("QA Chain built successfully.")
    return qa_chain

# -----------------------------------------------------------------------------
# 2) Interactive prompt (No changes needed here)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    qa_chain = build_chain()
    
    print("\n--- AI Second Brain is Ready ---")
    while True:
        query = input("\nYour question (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        print("Thinking...")
        output = qa_chain.invoke({"query": query})

        print("\n=== Answer ===")
        print(output["result"])
        
        if output.get("source_documents"):
            print("\n=== Sources ===")
            for doc in output["source_documents"]:
                m = doc.metadata
                source_type = m.get('source_type', 'N/A')
                source_file = m.get('source_file', 'N/A')
                page_seg = m.get('page_or_segment', 'N/A')
                print(f"- Type: {source_type}, File: {source_file}, Segment: {page_seg}")
