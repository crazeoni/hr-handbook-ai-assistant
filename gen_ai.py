from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
import os
import cohere
from dotenv import load_dotenv


load_dotenv()
REQUIRED_ENV_VARS = [
    "COHERE_API_KEY",
    "PINECONE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
]

# Fetch and validate all at once
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# If all are present, assign them for easy reference
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ✅ Set environment variable before creating vector store
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

with open("hr_document.txt", "r", encoding="utf-8") as file:
    hr_text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([hr_text])

# ---- Cohere embeddings ----
co = cohere.Client(COHERE_API_KEY)

def cohere_embed_texts(texts):
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"  # 👈 REQUIRED for v3 models
    )
    return response.embeddings

#embeddings = OpenAIEmbeddings(
#    model = "text-embedding-3-small",
#    api_key=OPENAI_API_KEY
#    )

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index if it doesn't exist yet
#index_name = "my-hr-index"
#if index_name not in [index["name"] for index in pc.list_indexes()]:
#    pc.create_index(
#        name=index_name,
#        dimension=1024,  # dimension for OpenAI embeddings
#        metric="cosine",
#        spec=ServerlessSpec(cloud="aws", region="us-east-1")
#    )

index_name = "my-hr-index-cohere"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- create vector store manually ----
print("📦 Generating embeddings for HR document...")
vectors = []
for i, doc in enumerate(docs):
    emb = cohere_embed_texts([doc.page_content])[0]
    vectors.append({
        "id": f"doc-{i}",
        "values": emb,
        "metadata": {"text": doc.page_content}
    })

print(f"✅ Created {len(vectors)} document embeddings!")

# Create vector store and add documents
#vectorstore = PineconeVectorStore.from_documents(
#    embedding=embeddings,
#    index_name=index_name
#)

#print("✅ Pinecone vector store created successfully!")

# ---- upsert to Pinecone ----
index.upsert(vectors=vectors)
print("✅ Pinecone vector store created successfully!")

query = "How many vacation days do I get?"
query_vector = cohere_embed_texts([query])[0]
#query_vector = embeddings.embed_query(query)

#results = index.query(query_vector, top_k=3, include_metadata=True)

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)


print(results)
print("\n🔍 Query Results:")
for match in results["matches"]:
    print(f"- Score: {match['score']:.4f}")
    print(f"  Text: {match['metadata']['text'][:200]}...\n")