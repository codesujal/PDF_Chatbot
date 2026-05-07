## PDF Chatbot

### Output:

![1](/Output/1.png)
![2](/Output/2.png)
![3](/Output/3.png)

### Diagram:

![Diagram](/diagram.png)

## How RAG works here:
### 1. PDF → Text
The uploaded PDF is read and converted into documents.
### 2. Chunking
Text is split into small overlapping chunks so it can be searched efficiently.
### 3. Embeddings
Each chunk is converted into a vector using HuggingFace embeddings.
### 4. Vector Store (Chroma)
All chunk vectors are stored in a database for semantic search.
### 5. Retrieval
When the user asks a question, it is also embedded.
Chroma finds the most similar chunks from the PDF.
### 6. Augmented Prompt
Retrieved chunks are combined as context with the user question.
### 7. Generation (Gemini)
The LLM answers using only this context.

## Tech Stack Used
### Frontend
* Streamlit – UI for file upload and chat interface

### Backend / Logic
* Python – core application logic
* LangChain – builds and manages the RAG pipeline

### Document Processing
* PyPDFLoader – extracts text from PDF
* RecursiveCharacterTextSplitter – splits text into chunks

### Embeddings
* HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) – converts text into vector embeddings

### Vector Database
* ChromaDB – stores embeddings and performs similarity search

### LLM (Answer Generation)
* Google Gemini (`ChatGoogleGenerativeAI`) – generates responses based on retrieved context

### Session Management
* Streamlit Session State – stores chat history during runtime
