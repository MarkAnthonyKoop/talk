#!/bin/bash

# Script to install the talk CLI project
# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Installing Talk CLI Project ===${NC}"
echo ""

# Check if Python 3.12 is available (or any Python 3.9+)
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo -e "${GREEN}Using Python 3.12${NC}"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}Using Python 3.11${NC}"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}Using Python 3.10${NC}"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo -e "${GREEN}Using Python 3.9${NC}"
else
    echo -e "${RED}Error: No suitable Python version found (need 3.9+)${NC}"
    exit 1
fi

# Show Python version
echo "Python version: $($PYTHON_CMD --version)"
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo -e "${RED}Error: setup.py not found!${NC}"
    echo "Please run this script from your project root"
    exit 1
fi

# Verify expected structure
if [ ! -d "talk" ] || [ ! -f "talk/talk.py" ]; then
    echo -e "${RED}Error: Expected to find talk/talk.py${NC}"
    echo "Current structure:"
    echo "  - setup.py: $([ -f setup.py ] && echo "û found" || echo "? missing")"
    echo "  - talk/: $([ -d talk ] && echo "û found" || echo "? missing")"
    echo "  - talk/talk.py: $([ -f talk/talk.py ] && echo "û found" || echo "? missing")"
    exit 1
fi

echo -e "${GREEN}û Project structure verified${NC}"

# Create virtual environment (optional)
echo -e "\n${YELLOW}Virtual environment setup${NC}"
echo -e "Do you want to create a virtual environment? (recommended) [y/N]"
read -r CREATE_VENV

if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment already exists. Use existing? [y/N]${NC}"
        read -r USE_EXISTING
        if [[ ! "$USE_EXISTING" =~ ^[Yy]$ ]]; then
            echo "Removing old virtual environment..."
            rm -rf venv
            $PYTHON_CMD -m venv venv
        fi
    else
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    echo -e "${GREEN}û Virtual environment activated${NC}"
    
    # Upgrade pip in venv
    pip install --upgrade pip setuptools wheel
fi

# Fix distutils issue for Python 3.12
echo -e "\n${YELLOW}Checking for distutils...${NC}"
if ! $PYTHON_CMD -c "import distutils" 2>/dev/null; then
    echo "Installing distutils for Python 3.12..."
    sudo apt install -y python3-distutils python3.12-distutils 2>/dev/null || {
        echo "Upgrading setuptools as fallback..."
        $PYTHON_CMD -m pip install --upgrade setuptools
    }
fi

# Install the package in development mode
echo -e "\n${YELLOW}Installing talk in development mode...${NC}"
pip install -e .

# Check if user wants to install optional dependencies
echo -e "\n${YELLOW}Install optional LLM backends?${NC}"
echo "1) Core only (minimal installation)"
echo "2) OpenAI support"
echo "3) Anthropic support" 
echo "4) Google Gemini support"
echo "5) All LLM backends"
echo "6) All + Development tools"
echo ""
echo -n "Choose [1-6]: "
read -r DEPS_CHOICE

case $DEPS_CHOICE in
    2)
        echo "Installing OpenAI support..."
        pip install -e ".[openai]"
        ;;
    3)
        echo "Installing Anthropic support..."
        pip install -e ".[anthropic]"
        ;;
    4)
        echo "Installing Gemini support..."
        pip install -e ".[gemini]"
        ;;
    5)
        echo "Installing all LLM backends..."
        pip install -e ".[all]"
        ;;
    6)
        echo "Installing everything including dev tools..."
        pip install -e ".[all,dev]"
        ;;
    *)
        echo "Installing core dependencies only..."
        ;;
esac

# Create .env template if needed and user doesn't have env vars
echo -e "\n${YELLOW}Checking environment variables...${NC}"
HAS_KEYS=false
if [ ! -z "$OPENAI_API_KEY" ] || [ ! -z "$ANTHROPIC_API_KEY" ] || [ ! -z "$GOOGLE_API_KEY" ]; then
    echo -e "${GREEN}û Found API keys in environment${NC}"
    HAS_KEYS=true
fi

if [ ! -f ".env" ] && [ "$HAS_KEYS" = false ]; then
    echo -e "${YELLOW}No API keys found in environment. Create .env template? [y/N]${NC}"
    read -r CREATE_ENV
    if [[ "$CREATE_ENV" =~ ^[Yy]$ ]]; then
        cat > .env << 'EOF'
# API Keys for LLM providers
# Uncomment and add your keys as needed

# OpenAI
# OPENAI_API_KEY=your-openai-api-key-here

# Anthropic  
# ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Google Gemini
# GOOGLE_API_KEY=your-google-api-key-here

# Other configuration
LOG_LEVEL=INFO
EOF
        echo -e "${GREEN}û Created .env template${NC}"
    fi
fi

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"

# Check if talk command is available
if command -v talk &> /dev/null; then
    echo -e "${GREEN}û 'talk' command is installed!${NC}"
    echo "Location: $(which talk)"
else
    echo -e "${YELLOW}? 'talk' command not found in PATH yet${NC}"
    echo "Try: source ~/.bashrc"
    echo "Or use: python -m talk.talk"
fi

# Test import
echo -e "\nTesting imports..."
$PYTHON_CMD -c "
try:
    import talk.talk
    print('û talk.talk imports successfully')
    # Check for main function
    if hasattr(talk.talk, 'main'):
        print('û main() function found')
    else:
        print('? main() function not found in talk.talk')
except Exception as e:
    print(f'? Import error: {e}')
"

# Show summary
echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo "Project: $(pwd)"
echo "Python: $($PYTHON_CMD --version)"

if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Virtual environment:${NC}"
    echo "  Activate with: source venv/bin/activate"
    echo "  Deactivate with: deactivate"
fi

echo -e "\n${BLUE}Usage examples:${NC}"
echo '  talk --task "Create a fibonacci function with tests"'
echo '  talk --task "Build a REST API for todos" --dir my_api'
echo '  talk --interactive --task "Analyze and optimize this code"'
echo '  talk --model gpt-4o-mini --task "Refactor this module"'
echo '  talk --help  # See all options'

echo -e "\n${GREEN}Ready to use! ??${NC}"

# Final check and tip
if ! command -v talk &> /dev/null 2>&1; then
    echo -e "\n${YELLOW}If 'talk' command still not found:${NC}"
    echo "1. source ~/.bashrc"
    echo "2. Or run directly: python -m talk.talk --help"
fi
