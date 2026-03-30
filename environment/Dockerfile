# AQDnet Docker Image
# Based on koji11235/aqdnet_env with added aqdnet package installation

FROM koji11235/aqdnet_env:latest

LABEL maintainer="koji11235"
LABEL description="AQDnet: Deep Neural Network for Protein-Ligand Docking and Scoring"

WORKDIR /workspace

# Copy repository files
COPY . /workspace/

# Install aqdnet package in editable mode
RUN pip install -e . --quiet && \
    echo "AQDnet package installed successfully"

# Set default command
CMD ["bash"]
