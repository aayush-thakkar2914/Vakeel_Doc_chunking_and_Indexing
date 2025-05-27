import os
import re
import json
from openai import OpenAI
from opensearchpy import OpenSearch, helpers

# === Configuration ===
GROQ_API_KEY = "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"  # Your GROQ API key
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
OPENSEARCH_PASS = "Avt@170291#$@"
INDEX_NAME = "chunked_text"

# === Step 1: Agentic Chunking using Groq API ===

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

text = """
OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence benefits all of humanity.
We're a team of researchers, engineers, and policy experts pushing the boundaries of what AI can do, while working to ensure it's safe and aligned with human values.
"""

prompt = f"""
You are a helpful assistant skilled at structuring and chunking content.
Break the following text into meaningful, coherent chunks (max 100-150 words each).
Return them as a JSON list with keys: "chunk_id", "content".

TEXT:
{text}
"""

response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {"role": "system", "content": "You are an expert document chunking assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
)

chunk_json = response.choices[0].message.content

# === Step 2: Extract and clean JSON from Groq response ===

print("Raw chunk_json response:")
print(chunk_json)  # Debug print to see exact response

# Attempt to extract JSON array from the response string (handles markdown/code fences or extra text)
match = re.search(r"\[\s*{.*}\s*\]", chunk_json, re.DOTALL)
if match:
    json_text = match.group(0)
else:
    # If no match found, fallback to the entire response
    json_text = chunk_json.strip()

# Clean common JSON issues:
# Remove trailing commas before ] or }
json_text = re.sub(r",(\s*[\]}])", r"\1", json_text)

# Strip leading/trailing whitespace
json_text = json_text.strip()

print("Cleaned JSON text to parse:")
print(json_text)  # Debug print

try:
    chunks = json.loads(json_text)
    if not isinstance(chunks, list):
        raise ValueError("Expected a list of chunk objects.")
except Exception as e:
    raise ValueError("Error parsing the chunked content from Groq. Make sure it's valid JSON.") from e

# === Step 3: Connect to OpenSearch and index chunks ===

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,           # Changed to True since OpenSearch uses HTTPS by default
    verify_certs=False,     # Disable certificate verification for local development
    ssl_show_warn=False,    # Suppress SSL warnings for local development
    timeout=30,             # Add timeout to prevent hanging connections
    max_retries=3,          # Add retry logic
    retry_on_timeout=True
)

# Test the connection first
try:
    info = opensearch_client.info()
    print("Successfully connected to OpenSearch!")
    print(f"Cluster info: {info}")
except Exception as e:
    print(f"Failed to connect to OpenSearch: {e}")
    exit(1)

# Create index if it does not exist
try:
    if not opensearch_client.indices.exists(index=INDEX_NAME):
        # Create index with proper mapping
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "content": {"type": "text"}
                }
            }
        }
        opensearch_client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists")
except Exception as e:
    print(f"Error creating index: {e}")
    exit(1)

# Prepare bulk indexing actions
actions = [
    {
        "_index": INDEX_NAME,
        "_source": {
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"]
        }
    }
    for chunk in chunks
]

try:
    helpers.bulk(opensearch_client, actions)
    print("✅ Chunks chunked and indexed successfully into OpenSearch!")
except Exception as e:
    print(f"Error indexing documents: {e}")
    exit(1)