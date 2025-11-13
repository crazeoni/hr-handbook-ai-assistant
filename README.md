## HR Handbook AI Assistant

An AI-powered document intelligence tool that allows users to upload and query large HR handbooks or policy documents using Retrieval-Augmented Generation (RAG).
It automatically breaks long text into semantic chunks, embeds them using Cohere embeddings, and stores them in Pinecone vector database for efficient retrieval and contextual question answering.

### Features

RAG-based document querying — Combines vector search + LLM reasoning for accurate answers.

PDF/text document ingestion — Reads and processes HR or policy documents.

Semantic search — Retrieves the most relevant text passages from embeddings.

Cohere embeddings integration — High-quality multilingual embeddings for HR data.

OpenAI/Router support — Extendable for generating human-like summarized responses.

Extensible architecture — Easily adaptable for legal, policy, or compliance use cases.

### Tech Stack

Python 3.10+

LangChain — RAG orchestration and document management

Pinecone — Vector storage and similarity search

Cohere API — Text embedding model (embed-english-v3.0)

OpenAI API / OpenRouter — Optional for LLM-driven summarization

Streamlit (optional) — For a simple web UI

### Setup
git clone https://github.com/crazeoni/hr-handbook-ai-assistant.git
cd hr-handbook-ai-assistant
pip install -r requirements.txt


Add your environment variables in a .env file:

COHERE_API_KEY=your-cohere-key
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key
OPENROUTER_API_KEY=your-openrouter-key


Then run:

python app.py

📘 Example Query
User: "How many vacation days do I get per year?"
→ The assistant retrieves the relevant section from the HR handbook
→ Summarizes or answers based on the text context

📦 Output Example
📦 Generating embeddings for HR document...
✅ Created 56 document embeddings!
✅ Pinecone vector store created successfully!
🔍 Query Results:
- Score: 0.8834
  Text: "Full-time employees are entitled to 20 days of paid vacation annually..."

### Future Enhancements

Add Streamlit web interface for file upload and chat

Integrate LangChain agents for multi-document reasoning

Enable document summarization & insights extraction

### Author

Fortune Stone — Full Stack Developer & AI Integration Specialist
🔗 Portfolio Website
 | LinkedIn: https://www.linkedin.com/in/ozioma-isaiah-29a198174/
 | Upwork: https://www.upwork.com/freelancers/~01af91268146ace3fd?mp_source=share
 | Fiverr: https://www.fiverr.com/s/YRZ2GRl
