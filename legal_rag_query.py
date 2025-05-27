import json
import re
from openai import OpenAI
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import argparse

GROQ_API_KEY = "your_api_key"
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
OPENSEARCH_PASS = "Avt@170291#$@"
INDEX_NAME = "chunked_documents"

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded successfully!")

class LegalRAGSystem:
    def __init__(self, index_name=INDEX_NAME):
        self.index_name = index_name
        self.opensearch_client = self._setup_opensearch()
        self.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )
    
    def _setup_opensearch(self):
        """Initialize OpenSearch client"""
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        try:
            info = client.info()
            print(f"Connected to OpenSearch cluster: {info.get('cluster_name', 'Unknown')}")
            return client
        except Exception as e:
            print(f"Failed to connect to OpenSearch: {e}")
            raise
    
    def search_similar_chunks(self, query, top_k=5, include_text_search=True):
        """
        Search for similar chunks using vector similarity and optional text search
        """
        print(f"Searching for: {query}")
        
        
        query_embedding = embedding_model.encode([query])[0].tolist()
        # Performing query based on the context passed through the vector search and models knowledge
        if include_text_search:
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": top_k
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "filename"],
                                    "type": "best_fields",
                                    "boost": 0.5
                                }
                            }
                        ]
                    }
                },
                "_source": ["content", "filename", "source_document", "page_info", "chunk_id", "file_type"]
            }
        else:
        # Performing only vector search
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                },
                "_source": ["content", "filename", "source_document", "page_info", "chunk_id", "file_type"]
            }
        
        try:
            response = self.opensearch_client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "content": source.get("content", ""),
                    "filename": source.get("filename", ""),
                    "source_document": source.get("source_document", ""),
                    "page_info": source.get("page_info", ""),
                    "chunk_id": source.get("chunk_id", ""),
                    "file_type": source.get("file_type", ""),
                    "score": hit["_score"]
                })
            
            print(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []
    
    def generate_response(self, query, relevant_chunks, max_context_length=4000):
        """
        Generate response using Groq with relevant chunks as context
        """
        
        # Prepare context from relevant chunks
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            chunk_text = f"""
Document: {chunk['filename']}
{f"Page: {chunk['page_info']}" if chunk['page_info'] else ""}
Content: {chunk['content']}
---"""
            
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        
        # Create the prompt for legal Q&A
        system_prompt = """You are an expert legal assistant with deep knowledge of legal documents and procedures. Your role is to provide accurate, helpful, and professional legal information based on the provided legal document context.

Guidelines:
1. Base your responses primarily on the provided legal document context
2. Provide clear, well-structured answers using proper legal terminology
3. If the context doesn't contain sufficient information to fully answer the question, clearly state this limitation
4. Include relevant citations to specific documents or sections when available
5. For complex legal matters, recommend consulting with a qualified attorney
6. Be precise and avoid speculation beyond what's supported by the documents
7. Structure your response with clear headings and bullet points when appropriate

Remember: This is for informational purposes only and does not constitute legal advice."""

        user_prompt = f"""Based on the following legal document excerpts, please answer the user's question:

LEGAL DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the legal documents provided. If you need to reference specific documents or sections, please cite them clearly."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    def query(self, question, top_k=5, include_text_search=True, show_sources=True):
        """
        Main query method that combines search and generation
        """
        print("=" * 60)
        print("🏛️ LEGAL RAG SYSTEM")
        print("=" * 60)
        
        # Searching for relevant chunks
        relevant_chunks = self.search_similar_chunks(question, top_k, include_text_search)
        
        if not relevant_chunks:
            return "I couldn't find any relevant legal documents to answer your question. Please ensure documents are properly indexed."
        
        #Generate response
        print("\nGenerating response...")
        response = self.generate_response(question, relevant_chunks)
        
        #Formattineng the response given by the model
        print(f"\n📋 QUESTION: {question}")
        print("\n📖 ANSWER:")
        print("-" * 40)
        print(response)
        
        if show_sources:
            print("\nSOURCES:")
            print("-" * 40)
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"{i}. Document: {chunk['filename']}")
                if chunk['page_info']:
                    print(f"   {chunk['page_info']}")
                print(f"   Relevance Score: {chunk['score']:.3f}")
                print(f"   Preview: {chunk['content'][:150]}...")
                print()
        
        return {
            "question": question,
            "answer": response,
            "sources": relevant_chunks
        }
    
    def interactive_mode(self):
        """
        Interactive chat mode for continuous querying
        """
        print("=" * 60)
        print("🏛️ LEGAL RAG SYSTEM - INTERACTIVE MODE")
        print("=" * 60)
        print("Type your legal questions. Type 'quit' or 'exit' to stop.")
        print("Commands:")
        print("  - 'sources on/off': Toggle source display")
        print("  - 'hybrid on/off': Toggle hybrid search mode")
        print("  - 'help': Show this help message")
        print("=" * 60)
        
        show_sources = True
        hybrid_search = True
        
        while True:
            try:
                user_input = input("\n🔍 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using the Legal RAG System!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  - 'sources on/off': Toggle source display")
                    print("  - 'hybrid on/off': Toggle hybrid search mode")
                    print("  - 'quit/exit': Exit the system")
                    continue
                
                if user_input.lower().startswith('sources'):
                    if 'on' in user_input.lower():
                        show_sources = True
                        print("✅ Source display enabled")
                    elif 'off' in user_input.lower():
                        show_sources = False
                        print("❌ Source display disabled")
                    continue
                
                if user_input.lower().startswith('hybrid'):
                    if 'on' in user_input.lower():
                        hybrid_search = True
                        print("✅ Hybrid search enabled")
                    elif 'off' in user_input.lower():
                        hybrid_search = False
                        print("❌ Hybrid search disabled (vector search only)")
                    continue
                
                if not user_input:
                    continue
                
                # Process the query
                self.query(user_input, top_k=5, include_text_search=hybrid_search, show_sources=show_sources)
                
            except KeyboardInterrupt:
                print("\n\nThank you for using the Legal RAG System!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Legal RAG Query System")
    parser.add_argument("--index", default=INDEX_NAME, help=f"OpenSearch index name (default: {INDEX_NAME})")
    parser.add_argument("--query", "-q", help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of relevant chunks to retrieve (default: 5)")
    parser.add_argument("--no-sources", action="store_true", help="Don't show source information")
    parser.add_argument("--vector-only", action="store_true", help="Use vector search only (no text search)")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag_system = LegalRAGSystem(args.index)
    
    if args.interactive:
        rag_system.interactive_mode()
    elif args.query:
        rag_system.query(
            args.query, 
            top_k=args.top_k, 
            include_text_search=not args.vector_only,
            show_sources=not args.no_sources
        )
    else:
        print("🏛️ Legal RAG Query System")
        print("\nUsage examples:")
        print("  python legal_rag_query.py -q \"What are the contract termination clauses?\"")
        print("  python legal_rag_query.py --interactive")
        print("  python legal_rag_query.py -q \"Define force majeure\" --no-sources")
        print("  python legal_rag_query.py -q \"Employment law requirements\" --vector-only")
        print("\nOptions:")
        print("  --interactive, -i     : Start interactive chat mode")
        print("  --query, -q          : Single query to process")
        print("  --top-k              : Number of relevant chunks to retrieve")
        print("  --no-sources         : Don't show source information")
        print("  --vector-only        : Use vector search only")
        print("  --index              : Specify OpenSearch index name")

if __name__ == "__main__":
    main()
