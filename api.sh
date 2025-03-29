#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/saurrx/Wan2GP-api.git"
CONDA_ENV_NAME="wan2gp"
PYTHON_VERSION="3.10.9"
API_PORT="3000"
VIDEO_OUTPUT_DIR="outputs"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Wan2.1 GP API Server Setup Script${NC}"
echo -e "${BLUE}======================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if conda is installed
if ! command_exists conda; then
    echo -e "${RED}Error: Conda is not installed or not in PATH.${NC}"
    echo -e "${YELLOW}Please install Miniconda or Anaconda first:${NC}"
    echo -e "${YELLOW}https://docs.conda.io/en/latest/miniconda.html${NC}"
    exit 1
fi

# Check if git is installed
if ! command_exists git; then
    echo -e "${RED}Error: Git is not installed or not in PATH.${NC}"
    echo -e "${YELLOW}Please install Git first.${NC}"
    exit 1
fi

# Create and navigate to project directory
PROJECT_DIR="$HOME/wan2gp_api"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo -e "${GREEN}Working directory: ${PROJECT_DIR}${NC}"

# Clone the repository if it doesn't exist
if [ ! -d "Wan2GP-api" ]; then
    echo -e "${BLUE}Cloning Wan2GP-api repository...${NC}"
    git clone "$REPO_URL"
    echo -e "${GREEN}Repository cloned successfully.${NC}"
else
    echo -e "${YELLOW}Wan2GP-api directory already exists. Skipping clone.${NC}"
    echo -e "${YELLOW}To get the latest version, delete the directory and run this script again.${NC}"
fi

# Navigate to the repository directory
cd Wan2GP-api

# Create and force activate conda environment
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo -e "${BLUE}Creating conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION...${NC}"
    conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
    echo -e "${GREEN}Conda environment created successfully.${NC}"
else
    echo -e "${YELLOW}Conda environment '$CONDA_ENV_NAME' already exists. Recreating environment to ensure clean state.${NC}"
    conda env remove -n "$CONDA_ENV_NAME" -y
    conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
    echo -e "${GREEN}Conda environment recreated successfully.${NC}"
fi

# Force activate conda environment
echo -e "${BLUE}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Install PyTorch
echo -e "${BLUE}Installing PyTorch...${NC}"
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124

# Install API-specific requirements
echo -e "${BLUE}Installing API dependencies...${NC}"
if [ -f "api_req.txt" ]; then
    pip install -r api_req.txt
else
    echo -e "${YELLOW}api_req.txt not found. Falling back to requirements.txt${NC}"
    pip install -r requirements.txt
fi

# Install Sage Attention
echo -e "${BLUE}Installing Sage Attention...${NC}"
if [ ! -d "../SageAttention" ]; then
    cd ..
    git clone https://github.com/thu-ml/SageAttention
    cd SageAttention
    pip install -e .
    cd ../Wan2GP-api
else
    echo -e "${YELLOW}SageAttention directory already exists. Using existing installation.${NC}"
fi

# Install Flash Attention
echo -e "${BLUE}Installing Flash Attention...${NC}"
pip install flash-attn==2.7.2.post1

# Create output directory
mkdir -p "$VIDEO_OUTPUT_DIR"
echo -e "${GREEN}Created video output directory: $VIDEO_OUTPUT_DIR${NC}"

# Create a run script for easy future execution
cat > run_api_server.sh << EOF
#!/bin/bash
# Run API Server Script

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Set environment variables
export VIDEO_OUTPUT_DIR="$VIDEO_OUTPUT_DIR"

# Run API server
echo "Starting Wan2.1 GP API Server on port $API_PORT..."
python api_server.py
EOF

chmod +x run_api_server.sh

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${PURPLE}Starting API server...${NC}"

# Set the video output directory environment variable
export VIDEO_OUTPUT_DIR="$VIDEO_OUTPUT_DIR"

# Run the API server
python api_server.py