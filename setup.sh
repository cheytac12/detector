#!/bin/bash

echo "Setting Python version to 3.11..."
pyenv local 3.11 2>/dev/null

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

echo "Launching application..."
python3 app.py
