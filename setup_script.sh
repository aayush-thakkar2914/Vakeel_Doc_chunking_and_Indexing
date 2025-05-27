#!/bin/bash

# Legal RAG System Setup Script
echo "🏛️ Setting up Legal RAG System"
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv legal_rag_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source legal_rag_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    echo "Please install Docker Desktop or Docker Engine"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed."
    echo "Please install Docker Compose"
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Start OpenSearch cluster
echo "🚀 Starting OpenSearch cluster..."
docker-compose up -d

# Wait for OpenSearch to be ready
echo "⏳ Waiting for OpenSearch to be ready..."
sleep 30

# Check OpenSearch health
max_attempts=12
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -k -u admin:Avt@170291#$@ https://localhost:9200/_cluster/health &> /dev/null; then
        echo "✅ OpenSearch is ready!"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts - OpenSearch not ready yet..."
        sleep 10
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ OpenSearch failed to start properly"
    echo "Check the logs with: docker-compose logs"
    exit 1
fi

# Test the connection
echo "🔍 Testing OpenSearch connection..."
curl -k -u admin:Avt@170291#$@ https://localhost:9200/_cluster/health?pretty

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source legal_rag_env/bin/activate"
echo "2. Add your legal documents to a folder"
echo "3. Run the chunker: python enhanced_chunker.py /path/to/legal/documents/"
echo "4. Query your documents: python legal_rag_query.py --interactive"
echo ""
echo "📚 Example commands:"
echo "# Process documents"
echo "python enhanced_chunker.py ./legal_docs/"
echo ""
echo "# Interactive query mode"
echo "python legal_rag_query.py --interactive"
echo ""
echo "# Single query"
echo "python legal_rag_query.py -q \"What are the contract termination procedures?\""
echo ""
echo "🔧 Management commands:"
echo "# Stop OpenSearch"
echo "docker-compose down"
echo ""
echo "# View OpenSearch logs"
echo "docker-compose logs -f"
echo ""
echo "# OpenSearch Dashboard (optional)"
echo "# Access at: http://localhost:5601"
echo "# Username: admin, Password: Avt@170291#$@"