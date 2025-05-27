#!/usr/bin/env python3
"""
Demo script showing how to use the Legal RAG System
"""

import os
import time
from pathlib import Path
from legal_rag_query import LegalRAGSystem

def create_sample_legal_document():
    """Create a sample legal document for demonstration"""
    
    sample_content = """
SAMPLE EMPLOYMENT CONTRACT

Article 1: Employment Terms
The Employee shall be employed in the position of Software Engineer, reporting directly to the CTO. The employment shall commence on January 1, 2024, and shall continue until terminated in accordance with the provisions of this Agreement.

Article 2: Compensation and Benefits
The Employee shall receive a base salary of $75,000 per annum, payable in equal monthly installments. The Employee shall also be eligible for performance-based bonuses as determined by the Company's discretion.

Article 3: Termination Clauses
Either party may terminate this employment agreement with thirty (30) days written notice. In case of termination for cause, including but not limited to misconduct, violation of company policies, or breach of confidentiality, the employment may be terminated immediately without notice.

Article 4: Confidentiality and Non-Disclosure
The Employee acknowledges that during employment, they may have access to confidential information including trade secrets, client lists, and proprietary technology. The Employee agrees to maintain strict confidentiality of such information both during and after employment.

Article 5: Non-Compete Clause
For a period of twelve (12) months following termination of employment, the Employee agrees not to engage in any business that directly competes with the Company within the same geographic region.

Article 6: Intellectual Property
All work products, inventions, and intellectual property created by the Employee during the course of employment shall be the exclusive property of the Company.

Article 7: Dispute Resolution
Any disputes arising from this agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.

Article 8: Governing Law
This agreement shall be governed by and construed in accordance with the laws of the State of California.

Article 9: Severability
If any provision of this agreement is found to be unenforceable, the remainder of the agreement shall remain in full force and effect.

Article 10: Entire Agreement
This agreement constitutes the entire agreement between the parties and supersedes all prior negotiations, representations, or agreements relating to the subject matter herein.
"""
    
    # Create sample documents directory
    os.makedirs("sample_legal_docs", exist_ok=True)
    
    # Write sample document
    with open("sample_legal_docs/employment_contract.txt", "w") as f:
        f.write(sample_content)
    
    # Create another sample document
    contract_sample = """
SAMPLE SOFTWARE LICENSE AGREEMENT

1. LICENSE GRANT
Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software solely for Licensee's internal business purposes.

2. RESTRICTIONS
Licensee shall not: (a) modify, adapt, or create derivative works of the Software; (b) reverse engineer, disassemble, or decompile the Software; (c) distribute, sell, or transfer the Software to any third party.

3. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and continue for a period of one (1) year unless earlier terminated. Either party may terminate this Agreement upon thirty (30) days written notice.

4. FORCE MAJEURE
Neither party shall be liable for any failure or delay in performance under this Agreement due to circumstances beyond its reasonable control, including but not limited to acts of God, natural disasters, war, terrorism, or government regulations.

5. LIMITATION OF LIABILITY
In no event shall either party be liable for any indirect, incidental, special, or consequential damages arising out of or relating to this Agreement, regardless of the theory of liability.

6. INDEMNIFICATION
Each party shall indemnify and hold harmless the other party from any claims, damages, or expenses arising from any breach of this Agreement or violation of applicable law.

7. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of law provisions.

8. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties and may only be modified in writing signed by both parties.
"""
    
    with open("sample_legal_docs/software_license.txt", "w") as f:
        f.write(contract_sample)
    
    print("✅ Created sample legal documents in 'sample_legal_docs/' directory")
    return "sample_legal_docs"

def demo_document_processing():
    """Demonstrate document processing and chunking"""
    
    print("🏛️ LEGAL RAG SYSTEM DEMO")
    print("=" * 50)
    
    # Create sample documents
    docs_dir = create_sample_legal_document()
    
    print(f"\n📁 Sample documents created in: {docs_dir}")
    print("Documents:")
    for file in Path(docs_dir).glob("*.txt"):
        print(f"  - {file.name}")
    
    print("\n📋 To process these documents, run:")
    print(f"python enhanced_chunker.py {docs_dir}")
    
    return docs_dir

def demo_queries():
    """Demonstrate various types of legal queries"""
    
    print("\n🔍 SAMPLE QUERIES TO TRY:")
    print("=" * 50)
    
    sample_queries = [
        "What are the termination procedures in the employment contract?",
        "What is the notice period required for contract termination?",
        "What are the confidentiality obligations?",
        "Explain the non-compete clause duration and restrictions",
        "What happens to intellectual property created during employment?",
        "How are disputes resolved according to the contracts?",
        "What constitutes a force majeure event?",
        "What are the salary and compensation details?",
        "What are the liability limitations in the software license?",
        "What law governs these agreements?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"{i:2d}. {query}")
    
    print(f"\n📝 To run these queries interactively:")
    print("python legal_rag_query.py --interactive")
    
    print(f"\n🔍 To run a single query:")
    print("python legal_rag_query.py -q \"What are the termination procedures?\"")

def demo_interactive_session():
    """Demonstrate an interactive session (requires documents to be indexed)"""
    
    print("\n🤖 INTERACTIVE DEMO SESSION")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag_system = LegalRAGSystem()
        
        # Test queries
        test_queries = [
            "What are the main termination clauses?",
            "What is the notice period for termination?",
            "Explain the confidentiality requirements"
        ]
        
        print("Running sample queries...\n")
        
        for query in test_queries:
            print(f"🔍 Testing query: {query}")
            result = rag_system.query(query, top_k=3, show_sources=False)
            print("✅ Query processed successfully\n")
            time.sleep(1)  # Brief pause between queries
        
        print("🎉 Demo queries completed successfully!")
        print("\nFor full interactive mode, run:")
        print("python legal_rag_query.py --interactive")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nMake sure to:")
        print("1. Start OpenSearch: docker-compose up -d")
        print("2. Process documents: python enhanced_chunker.py sample_legal_docs/")
        print("3. Then run this demo again")

def main():
    """Main demo function"""
    
    print("🏛️ LEGAL RAG SYSTEM - COMPLETE DEMO")
    print("=" * 60)
    
    while True:
        print("\nChoose a demo option:")
        print("1. Create sample documents and show processing commands")
        print("2. Show sample queries")
        print("3. Run interactive demo (requires indexed documents)")
        print("4. Show complete setup instructions")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            demo_document_processing()
        
        elif choice == "2":
            demo_queries()
        
        elif choice == "3":
            demo_interactive_session()
        
        elif choice == "4":
            show_setup_instructions()
        
        elif choice == "5":
            print("Thank you for trying the Legal RAG System demo!")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

def show_setup_instructions():
    """Show complete setup instructions"""
    
    print("\n📋 COMPLETE SETUP INSTRUCTIONS")
    print("=" * 50)
    
    instructions = """
1. PREREQUISITES:
   - Python 3.8 or higher
   - Docker and Docker Compose
   - At least 4GB RAM available for OpenSearch

2. INSTALLATION:
   chmod +x setup.sh
   ./setup.sh

3. MANUAL SETUP (if script fails):
   # Create virtual environment
   python3 -m venv legal_rag_env
   source legal_rag_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start OpenSearch
   docker-compose up -d
   
   # Wait for OpenSearch to be ready (30-60 seconds)

4. PROCESS DOCUMENTS:
   python enhanced_chunker.py /path/to/legal/documents/
   
   # Or use sample documents
   python enhanced_chunker.py sample_legal_docs/

5. QUERY DOCUMENTS:
   # Interactive mode
   python legal_rag_query.py --interactive
   
   # Single query
   python legal_rag_query.py -q "Your legal question here"

6. TROUBLESHOOTING:
   # Check OpenSearch status
   curl -k -u admin:Avt@170291#$@ https://localhost:9200/_cluster/health
   
   # View logs
   docker-compose logs -f
   
   # Restart OpenSearch
   docker-compose down && docker-compose up -d

7. FEATURES:
   ✅ PDF and text document support
   ✅ Intelligent legal document chunking
   ✅ Vector similarity search
   ✅ Hybrid search (vector + text)
   ✅ Context-aware legal Q&A
   ✅ Source citation and references
   ✅ Interactive chat mode
"""
    
    print(instructions)

if __name__ == "__main__":
    main()