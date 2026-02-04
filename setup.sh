#!/bin/bash

# AI Voice Detection - Quick Start Script
# This script guides you through the entire setup process

set -e  # Exit on error

echo "=================================================="
echo "🎯 AI Voice Detection System - Quick Start"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Python is installed${NC}"
echo ""

# Step 1: Install dependencies
echo "=================================================="
echo "Step 1: Installing Dependencies"
echo "=================================================="
echo ""

read -p "Install Python dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo "Skipped dependency installation"
fi
echo ""

# Step 2: Dataset collection
echo "=================================================="
echo "Step 2: Dataset Collection"
echo "=================================================="
echo ""

if [ ! -d "data" ]; then
    echo "Creating data directory structure..."
    python3 collect_dataset.py
else
    echo "Data directory already exists"
    read -p "Run dataset collection tool? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 collect_dataset.py
    fi
fi
echo ""

# Step 3: Model training
echo "=================================================="
echo "Step 3: Model Training"
echo "=================================================="
echo ""

if [ ! -f "voice_classifier_best.pth" ]; then
    echo -e "${YELLOW}⚠️  Model file not found${NC}"
    echo ""
    echo "To train the model, you need:"
    echo "1. Human voice samples in data/human/[language]/"
    echo "2. AI-generated samples in data/ai_generated/[language]/"
    echo ""
    
    # Check if data exists
    human_count=$(find data/human -type f 2>/dev/null | wc -l)
    ai_count=$(find data/ai_generated -type f 2>/dev/null | wc -l)
    
    echo "Current dataset:"
    echo "  Human samples: $human_count"
    echo "  AI samples: $ai_count"
    echo ""
    
    if [ $human_count -gt 0 ] && [ $ai_count -gt 0 ]; then
        read -p "Start training now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Starting training..."
            python3 train_model.py
            echo -e "${GREEN}✓ Model training complete${NC}"
        else
            echo "Skipped training"
            echo "Run 'python3 train_model.py' when ready"
        fi
    else
        echo -e "${YELLOW}⚠️  Dataset is empty or incomplete${NC}"
        echo "Please collect dataset first (see README.md)"
    fi
else
    echo -e "${GREEN}✓ Model file exists: voice_classifier_best.pth${NC}"
    
    read -p "Retrain model? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 train_model.py
    fi
fi
echo ""

# Step 4: Configure API
echo "=================================================="
echo "Step 4: API Configuration"
echo "=================================================="
echo ""

echo "Current API configuration:"
echo "  Default API Key: your-secure-api-key-here"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Change the API key before deployment!${NC}"
echo ""

read -p "Generate a secure API key now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    new_key=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo ""
    echo "Generated API Key:"
    echo -e "${GREEN}$new_key${NC}"
    echo ""
    echo "To use this key, update the following line in api.py:"
    echo "  VALID_API_KEY = \"$new_key\""
    echo ""
    read -p "Press Enter to continue..."
fi
echo ""

# Step 5: Test locally
echo "=================================================="
echo "Step 5: Local Testing"
echo "=================================================="
echo ""

if [ -f "voice_classifier_best.pth" ]; then
    echo "Starting API server..."
    echo ""
    echo "The API will start at: http://localhost:8000"
    echo ""
    echo "In another terminal, run:"
    echo "  python3 test_api.py"
    echo ""
    
    read -p "Start API server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting server..."
        echo "Press Ctrl+C to stop"
        python3 api.py
    else
        echo "Skipped server start"
        echo "Run 'python3 api.py' to start the server manually"
    fi
else
    echo -e "${RED}❌ Cannot start API: Model file not found${NC}"
    echo "Please train the model first"
fi
echo ""

# Step 6: Deployment
echo "=================================================="
echo "Step 6: Deployment"
echo "=================================================="
echo ""

echo "Deployment options:"
echo ""
echo "1. Railway (Recommended - Easiest)"
echo "   Commands:"
echo "     railway login"
echo "     railway init"
echo "     railway up"
echo "     railway domain"
echo ""
echo "2. Render.com (Free tier)"
echo "   Visit: https://render.com"
echo ""
echo "3. Google Cloud Run"
echo "   Command: gcloud run deploy"
echo ""
echo "4. Docker (Any platform)"
echo "   Commands:"
echo "     docker build -t voice-detector ."
echo "     docker run -p 8000:8000 voice-detector"
echo ""

read -p "View detailed deployment guide? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat DEPLOYMENT_GUIDE.md | less
fi
echo ""

# Final checklist
echo "=================================================="
echo "📋 Pre-Submission Checklist"
echo "=================================================="
echo ""

checks=(
    "voice_classifier_best.pth:Model trained"
    "api.py:API code ready"
    "requirements.txt:Dependencies listed"
    "Dockerfile:Docker configuration"
)

for check in "${checks[@]}"; do
    IFS=':' read -r file desc <<< "$check"
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $desc ($file)"
    else
        echo -e "${RED}✗${NC} $desc ($file) - MISSING"
    fi
done

echo ""
echo "Before submission, ensure:"
echo "  [ ] API is deployed and publicly accessible"
echo "  [ ] API key is configured"
echo "  [ ] Health endpoint returns 200"
echo "  [ ] Detection endpoint works correctly"
echo "  [ ] All 5 languages are supported"
echo "  [ ] Response format matches specification"
echo "  [ ] Latency is < 2 seconds"
echo ""

echo "=================================================="
echo "🎉 Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Test your API thoroughly"
echo "2. Deploy to production"
echo "3. Submit your endpoint + API key"
echo "4. Monitor during evaluation"
echo ""
echo "Good luck! 🚀"
echo ""
