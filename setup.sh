c#!/bin/bash

# URL of the Miniconda installer
#URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Output file name
#OUTPUT="Miniconda3-latest-Linux-x86_64.sh"

# Download the file using curl
#curl -o "$OUTPUT" "$URL"

# Make the script executable
#chmod +x "$OUTPUT"

conda create -n src_rp python=3.12
conda activate 
conda install jupyter
pip install -r requirements.txt
ipython kernel install --user --name=src_rp
