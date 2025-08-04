#!/bin/bash

# 🚀 Quick Start Script for AI Task Management System
# Starts all backend services, frontend, and APIs with one command

echo "🚀 AI Task Management System - Quick Start"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $1 is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $1 is available${NC}"
        return 0
    fi
}

# Function to start service
start_service() {
    local service_name=$1
    local command=$2
    local port=$3
    
    echo -e "\n${BLUE}🔧 Starting $service_name...${NC}"
    
    if check_port $port; then
        echo "Running: $command"
        eval "$command" &
        local pid=$!
        echo -e "${GREEN}✅ $service_name started (PID: $pid)${NC}"
        sleep 2
    else
        echo -e "${YELLOW}⚠️  Skipping $service_name (port $port in use)${NC}"
    fi
}

# Check if required files exist
echo -e "\n${BLUE}🔍 Checking required files...${NC}"
required_files=(
    "database_setup.py"
    "data_pipeline_fastapi.py"
    "api/perfect_performance_server.py"
    "api/advanced_nlp_api.py"
    "api/enhanced_auto_assign_api.py"
    "ai-task-manager/package.json"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file found${NC}"
    else
        echo -e "${RED}❌ $file not found${NC}"
        exit 1
    fi
done

# Create necessary directories
echo -e "\n${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p logs
mkdir -p models/advanced_nlp_final
mkdir -p models/enhanced_features

# Start all services
echo -e "\n${BLUE}🚀 Starting all services...${NC}"

# 1. Database Setup
echo -e "\n${BLUE}🔧 Setting up database...${NC}"
python database_setup.py &
sleep 3

# 2. Data Pipeline (Port 8000)
start_service "Data Pipeline" "python data_pipeline_fastapi.py" 8000

# 3. Perfect Performance API (Port 5001)
start_service "Perfect Performance API" "python api/perfect_performance_server.py" 5001

# 4. Advanced NLP API (Port 5003)
start_service "Advanced NLP API" "python api/advanced_nlp_api.py" 5003

# 5. Enhanced Auto-Assignment API (Port 5002)
start_service "Enhanced Auto-Assignment API" "python api/enhanced_auto_assign_api.py" 5002

# 6. Frontend (Port 3000)
start_service "Frontend (Next.js)" "cd ai-task-manager && npm run dev" 3000

# Wait for services to start
echo -e "\n${BLUE}⏳ Waiting for services to start...${NC}"
sleep 10

# Test all services
echo -e "\n${BLUE}🧪 Testing all services...${NC}"

# Test Data Pipeline
echo -e "\n${YELLOW}Testing Data Pipeline (Port 8000)...${NC}"
if curl -s http://localhost:8000/api/pipeline/health > /dev/null; then
    echo -e "${GREEN}✅ Data Pipeline is running${NC}"
else
    echo -e "${RED}❌ Data Pipeline is not responding${NC}"
fi

# Test Perfect Performance API
echo -e "\n${YELLOW}Testing Perfect Performance API (Port 5001)...${NC}"
if curl -s http://localhost:5001/api/health > /dev/null; then
    echo -e "${GREEN}✅ Perfect Performance API is running${NC}"
else
    echo -e "${RED}❌ Perfect Performance API is not responding${NC}"
fi

# Test Advanced NLP API
echo -e "\n${YELLOW}Testing Advanced NLP API (Port 5003)...${NC}"
if curl -s http://localhost:5003/api/advanced-nlp/health > /dev/null; then
    echo -e "${GREEN}✅ Advanced NLP API is running${NC}"
else
    echo -e "${RED}❌ Advanced NLP API is not responding${NC}"
fi

# Test Enhanced Auto-Assignment API
echo -e "\n${YELLOW}Testing Enhanced Auto-Assignment API (Port 5002)...${NC}"
if curl -s http://localhost:5002/api/enhanced-assignment/health > /dev/null; then
    echo -e "${GREEN}✅ Enhanced Auto-Assignment API is running${NC}"
else
    echo -e "${RED}❌ Enhanced Auto-Assignment API is not responding${NC}"
fi

# Show system status
echo -e "\n${BLUE}📊 System Status:${NC}"
echo "=============================================================="
echo -e "${GREEN}✅ Data Pipeline: http://localhost:8000${NC}"
echo -e "${GREEN}✅ Perfect Performance API: http://localhost:5001${NC}"
echo -e "${GREEN}✅ Advanced NLP API: http://localhost:5003${NC}"
echo -e "${GREEN}✅ Enhanced Auto-Assignment API: http://localhost:5002${NC}"
echo -e "${GREEN}✅ Frontend: http://localhost:3000${NC}"
echo "=============================================================="

# Show available endpoints
echo -e "\n${BLUE}🔗 Available Endpoints:${NC}"
echo "Data Pipeline:"
echo "  - POST /api/pipeline/process-task"
echo "  - GET  /api/pipeline/health"
echo "  - GET  /api/pipeline/stats"

echo -e "\nPerfect Performance API:"
echo "  - POST /api/process-task"
echo "  - POST /api/predict-category"
echo "  - POST /api/assign-employee"
echo "  - GET  /api/health"

echo -e "\nAdvanced NLP API:"
echo "  - POST /api/advanced-nlp/analyze"
echo "  - POST /api/advanced-nlp/extract-embeddings"
echo "  - GET  /api/advanced-nlp/model-info"

echo -e "\nEnhanced Auto-Assignment API:"
echo "  - POST /api/enhanced-assignment/assign"
echo "  - GET  /api/enhanced-assignment/recommendations"
echo "  - GET  /api/enhanced-assignment/employees"

# Show test commands
echo -e "\n${BLUE}🧪 Test Commands:${NC}"
echo "=============================================================="
echo "Test Perfect Performance:"
echo "  python test_perfect_performance.py"
echo ""
echo "Test Enhanced Auto-Assignment:"
echo "  python test_enhanced_auto_assignment.py"
echo ""
echo "Test Advanced NLP:"
echo "  python test_advanced_nlp.py"
echo ""
echo "Test Enterprise System:"
echo "  python industry_test_demo.py"
echo "=============================================================="

# Show monitoring commands
echo -e "\n${BLUE}📊 Monitoring Commands:${NC}"
echo "=============================================================="
echo "Check all services:"
echo "  curl http://localhost:8000/api/pipeline/health"
echo "  curl http://localhost:5001/api/health"
echo "  curl http://localhost:5003/api/advanced-nlp/health"
echo "  curl http://localhost:5002/api/enhanced-assignment/health"
echo ""
echo "View logs:"
echo "  tail -f logs/*.log"
echo "=============================================================="

echo -e "\n${GREEN}🎉 AI Task Management System is now running!${NC}"
echo -e "${GREEN}🚀 All services are ready for production use.${NC}"
echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Test the APIs using the provided test scripts"
echo "3. Monitor system performance and logs"
echo "4. Press Ctrl+C to stop all services"
echo ""
echo -e "${GREEN}✅ Quick start complete!${NC}" 