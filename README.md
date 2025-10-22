# GeoRAG: A Geography-Focused RAG System

## What This Is

I built this RAG system because I'm interested in both geography and AI technology. The system lets me ask questions about US geography and get answers based on actual documents I've processed and learn more about the diverse climate and biomes of the US, while also exploring AI technology. 


## Technical Implementation

### Document Processing (`db_creation.py`)
- Uses **LangChain** to load PDFs and markdown files
- Implements **RecursiveCharacterTextSplitter** with 1000-character chunks and 200-character overlap
- Creates embeddings using **OpenAI's text-embedding-ada-002**
- Stores everything in **ChromaDB** for fast similarity search

### Query Processing (`create_response.py`)
- Takes a question via command line argument
- Searches the ChromaDB for the 3 most relevant document chunks
- Combines the context and sends it to **GPT-3.5-turbo** with a structured prompt
- Returns the AI-generated answer based on the retrieved context

## Technologies Used

- **LangChain** - For document loading, text splitting, and vector store integration
- **ChromaDB** - Vector database for storing and searching embeddings
- **OpenAI API** - For embeddings and text generation
- **PyPDF** - PDF document processing
- **Python** - Core implementation language

## How to Run It

1. **Set up your environment:**
   ```bash
   pip install -r requirements.txt
   pip install "unstructured[md]"
   ```

2. **Add your OpenAI API key to a `.env.local` file:**
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. **Process the documents:**
   ```bash
   python db_creation.py
   ```

4. **Ask questions:**
   ```bash
   python create_response.py "What is the highest peak in the United States?"
   ```


## Technical Details

- **Chunk size**: 1000 characters with 200-character overlap
- **Retrieval**: Top 3 most similar chunks for context
- **Embeddings**: OpenAI's text-embedding-ada-002
- **LLM**: GPT-3.5-turbo
- **Storage**: ChromaDB with SQLite backend

## Code Structure

- `db_creation.py` - Handles document loading, chunking, and vector database creation
- `create_response.py` - Takes queries, searches the database, and generates responses
- `Data/` - Contains the source documents (PDF and markdown files)
- `chroma/` - The vector database storage

Fun way to learn about RAG systems while working with content I actually care about!
