#!/bin/bash

# Create a virtual environment if not exist
ENV_NAME="env"
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"
fi

# Activate the env
echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping installation of dependencies."
fi

# Run the programs
echo "Running eda.py..."
python3 eda.py

echo "Running main.py..."
python3 main.py

# Deactivate env
echo "Deactivating virtual environment..."
deactivate
