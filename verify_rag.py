import os
import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. Configuration ---
# The path MUST be the same as in your build and verification scripts
db_path = os.path.join("data", "chroma_db")

# Load the same text embedding model you used to create the database
# This is crucial for getting accurate results.
print("Loading embedding model...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# --- 2. Connect to the Persistent Database ---
print(f"Connecting to ChromaDB at: {db_path}")
try:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="ai_second_brain")
    print("Successfully connected to the 'ai_second_brain' collection.")
except Exception as e:
    print(f"\n❌ Failed to connect to ChromaDB: {e}")
    exit() # Exit the script if connection fails

# --- 3. The Search Function ---
def search(query_text, n_results=5):
    """
    Searches the collection for a given query text.

    Args:
        query_text (str): The text to search for.
        n_results (int): The number of results to return.

    Returns:
        The results of the query.
    """
    if not query_text:
        print("Query cannot be empty.")
        return None

    print(f"\n🔍 Searching for: '{query_text}'...")

    # Create an embedding for the query text
    query_embedding = text_model.encode(query_text).tolist()

    # Perform the query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"] # Specify what to include in the results
    )

    return results

# --- 4. Example Usage ---
if __name__ == "__main__":
    # Check if the collection is empty before searching
    if collection.count() == 0:
        print("\n⚠️ The collection is empty. Please run your build script first.")
    else:
        # --- Perform a search ---
        # Change this query to whatever you want to find!
        my_query = "what is a transformer model"
        search_results = search(my_query, n_results=5)

        # --- Print the results ---
        if search_results:
            print("\n✅ Search complete. Results:\n" + "="*30)
            
            # The results are returned as a dictionary of lists, even for a single query.
            # We access the first (and only) list of results using [0].
            distances = search_results['distances'][0]
            metadatas = search_results['metadatas'][0]
            documents = search_results['documents'][0]

            for i in range(len(distances)):
                print(f"▶️  Result {i+1} (Distance: {distances[i]:.4f})")
                
                meta = metadatas[i]
                source_type = meta.get('source_type', 'N/A')
                source_file = meta.get('source_file', 'N/A')

                print(f"   Source Type: {source_type}")
                print(f"   Source File: {source_file}")

                # Display text preview or image path based on the source type
                if source_type == 'image':
                    print(f"   Image Path:  {meta.get('image_path', 'N/A')}")
                else:
                    # The 'documents' field contains the text chunk
                    print(f"   Content:     \"{documents[i]}...\"")
                print("-" * 20)