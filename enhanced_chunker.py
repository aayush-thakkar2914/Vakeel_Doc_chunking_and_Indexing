import os
import re
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import argparse
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyPDF2 not installed. PDF support disabled.")
    print("Install with: pip install PyPDF2")

# === Configuration ===
GROQ_API_KEY = "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"  # Your GROQ API key
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
OPENSEARCH_PASS = "Avt@170291#$@"
INDEX_NAME = "chunked_documents"

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.xml'}
if PDF_SUPPORT:
    SUPPORTED_EXTENSIONS.add('.pdf')

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded successfully!")

def read_pdf(file_path):
    """Extract text from PDF file"""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text.strip()
    
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")

def read_document(file_path):
    """Read document content from various file types"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Supported: {SUPPORTED_EXTENSIONS}")
    
    # Handle PDF files separately
    if file_path.suffix.lower() == '.pdf':
        try:
            content = read_pdf(file_path)
            return content, file_path.name
        except Exception as e:
            raise ValueError(f"Could not read PDF file {file_path}: {e}")
    
    # Handle text-based files
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content, file_path.name
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            return content, file_path.name
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")

def chunk_document(text, filename, max_chunk_size=150):
    """Use Groq API to chunk the document intelligently"""
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )

    prompt = f"""
    You are a helpful assistant skilled at structuring and chunking legal content.
    Break the following legal document into meaningful, coherent chunks (max {max_chunk_size} words each).
    
    Important guidelines for legal documents:
    - Maintain legal context and meaning within each chunk
    - Break at natural boundaries (sections, subsections, clauses, paragraphs)
    - Each chunk should be self-contained and legally meaningful
    - Include relevant section headers or clause numbers if present
    - Preserve important legal terminology and references
    
    Return the result as a JSON list with keys: "chunk_id", "content", "word_count".
    
    DOCUMENT FILENAME: {filename}
    DOCUMENT CONTENT:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert legal document chunking assistant specializing in legal texts. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent legal chunking
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error calling Groq API: {e}")

def generate_embeddings(chunks):
    """Generate vector embeddings for chunks using sentence-transformers"""
    
    print("Generating embeddings for chunks...")
    
    texts = []
    for chunk in chunks:
        content = chunk.get('content', '')
        # Clean content for better embeddings
        cleaned_content = re.sub(r"--- Page \d+ ---\n?", "", content).strip()
        texts.append(cleaned_content)
    
    # Generate embeddings
    embeddings = embedding_model.encode(texts)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    print(f"Generated embeddings for {len(chunks)} chunks")
    return chunks

def parse_chunks(chunk_json, filename):
    """Extract and clean JSON from Groq response"""
    
    print("Raw chunk_json response:")
    print(chunk_json[:500] + "..." if len(chunk_json) > 500 else chunk_json)
    
    # Try to extract JSON array from the response
    json_pattern = r"```json\s*(.*?)\s*```|```\s*(.*?)\s*```|\[\s*{.*?}\s*\]"
    match = re.search(json_pattern, chunk_json, re.DOTALL)
    
    if match:
        json_text = match.group(1) or match.group(2) or match.group(0)
    else:
        # If no code blocks found, look for JSON array
        match = re.search(r"\[\s*{.*}\s*\]", chunk_json, re.DOTALL)
        if match:
            json_text = match.group(0)
        else:
            json_text = chunk_json.strip()
    
    # Clean common JSON issues
    json_text = re.sub(r",(\s*[\]}])", r"\1", json_text)
    json_text = json_text.strip()
    
    print("Cleaned JSON text to parse:")
    print(json_text[:300] + "..." if len(json_text) > 300 else json_text)
    
    try:
        chunks = json.loads(json_text)
        if not isinstance(chunks, list):
            raise ValueError("Expected a list of chunk objects.")
        
        # Add filename to each chunk
        for chunk in chunks:
            chunk['filename'] = filename
            chunk['source_document'] = filename
            
        return chunks
    except Exception as e:
        raise ValueError(f"Error parsing chunks from Groq response: {e}")

def setup_opensearch_client():
    """Initialize and test OpenSearch connection"""
    
    opensearch_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True
    )
    
    # Test connection
    try:
        info = opensearch_client.info()
        print("Successfully connected to OpenSearch!")
        print(f"Cluster: {info.get('cluster_name', 'Unknown')}")
        return opensearch_client
    except Exception as e:
        print(f"Failed to connect to OpenSearch: {e}")
        print("Make sure OpenSearch is running and accessible.")
        raise

def create_index_if_not_exists(client, index_name):
    """Create index with proper mapping including vector field"""
    
    try:
        if not client.indices.exists(index=index_name):
            index_body = {
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "knn": True,  # Enable k-NN for vector search
                    }
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "filename": {"type": "keyword"},
                        "source_document": {"type": "keyword"},
                        "file_type": {"type": "keyword"},
                        "word_count": {"type": "integer"},
                        "page_info": {"type": "text"},
                        "indexed_at": {"type": "date"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 384,  # all-MiniLM-L6-v2 embedding dimension
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        }
                    }
                }
            }
            client.indices.create(index=index_name, body=index_body)
            print(f"Created index: {index_name}")
        else:
            print(f"Index {index_name} already exists")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def index_chunks(client, chunks, index_name, filename):
    """Index chunks into OpenSearch with embeddings"""
    
    from datetime import datetime
    
    # Determine file type
    file_extension = Path(filename).suffix.lower()
    file_type = file_extension[1:] if file_extension else 'unknown'
    
    actions = []
    for chunk in chunks:
        # Extract page information if present in content
        page_info = ""
        content = chunk.get("content", "")
        if "--- Page" in content:
            page_match = re.search(r"--- Page (\d+) ---", content)
            if page_match:
                page_info = f"Page {page_match.group(1)}"
                # Clean the content by removing page markers
                content = re.sub(r"--- Page \d+ ---\n?", "", content).strip()
        
        action = {
            "_index": index_name,
            "_source": {
                "chunk_id": chunk.get("chunk_id"),
                "content": content,
                "filename": chunk.get("filename"),
                "source_document": chunk.get("source_document"),
                "file_type": file_type,
                "word_count": chunk.get("word_count", 0),
                "page_info": page_info,
                "indexed_at": datetime.now().isoformat(),
                "embedding": chunk.get("embedding")
            }
        }
        actions.append(action)
    
    try:
        result = helpers.bulk(client, actions)
        print(f"Successfully indexed {len(chunks)} chunks with embeddings!")
        return result
    except Exception as e:
        print(f"Error indexing chunks: {e}")
        raise

def process_document(file_path, max_chunk_size=150):
    """Main function to process a single document"""
    
    print(f"\nProcessing document: {file_path}")
    
    # Step 1: Read document
    try:
        content, filename = read_document(file_path)
        print(f"Read document: {len(content)} characters")
    except Exception as e:
        print(f"Error reading document: {e}")
        return False
    
    # Step 2: Chunk document
    try:
        chunk_json = chunk_document(content, filename, max_chunk_size)
        chunks = parse_chunks(chunk_json, filename)
        print(f"Created {len(chunks)} chunks")
    except Exception as e:
        print(f"Error chunking document: {e}")
        return False
    
    # Step 3: Generate embeddings
    try:
        chunks = generate_embeddings(chunks)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return False
    
    # Step 4: Setup OpenSearch
    try:
        client = setup_opensearch_client()
        create_index_if_not_exists(client, INDEX_NAME)
    except Exception as e:
        print(f"Error setting up OpenSearch: {e}")
        return False
    
    # Step 5: Index chunks
    try:
        index_chunks(client, chunks, INDEX_NAME, filename)
        print(f"Document processing completed successfully!")
        return True
    except Exception as e:
        print(f"Error indexing chunks: {e}")
        return False

def process_directory(directory_path, max_chunk_size=150):
    """Process all supported documents in a directory"""
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return
    
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
    
    if not files:
        print(f"No supported files found in {directory_path}")
        print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"Found {len(files)} files to process")
    
    successful = 0
    for file_path in files:
        if process_document(file_path, max_chunk_size):
            successful += 1
    
    print(f"\nProcessing completed: {successful}/{len(files)} files successful")

def main():
    """Main CLI interface"""
    
    global INDEX_NAME

    parser = argparse.ArgumentParser(description="Chunk legal documents and index them in OpenSearch with vector embeddings")
    parser.add_argument("path", help="Path to document file or directory")
    parser.add_argument("--chunk-size", type=int, default=150, help="Maximum words per chunk (default: 150)")
    parser.add_argument("--index", default=INDEX_NAME, help=f"OpenSearch index name (default: {INDEX_NAME})")
    
    args = parser.parse_args()
    INDEX_NAME = args.index
    
    path = Path(args.path)
    
    if path.is_file():
        process_document(path, args.chunk_size)
    elif path.is_dir():
        process_directory(path, args.chunk_size)
    else:
        print(f"❌ Path not found: {args.path}")

if __name__ == "__main__":
    # Example usage if no command line arguments
    import sys
    
    if len(sys.argv) == 1:
        print("🏛️ Legal Document Chunking and Indexing Tool")
        print("\nUsage examples:")
        print("  python enhanced_chunker.py legal_document.pdf")
        print("  python enhanced_chunker.py /path/to/legal/documents/")
        print("  python enhanced_chunker.py contract.txt --chunk-size 200")
        print("  python enhanced_chunker.py legal_docs/ --index legal_index")
        print(f"\nSupported file types: {', '.join(SUPPORTED_EXTENSIONS)}")
        print("\nFeatures:")
        print("  ✅ Legal document chunking with context preservation")
        print("  ✅ Vector embeddings for semantic search")
        print("  ✅ OpenSearch indexing with k-NN support")
        print("  ✅ PDF support for legal documents")
    else:
        main()