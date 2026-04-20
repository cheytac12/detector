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
python3 -m pip install --upgrade pip > /dev/null

echo "Synchronizing dependencies..."
# This will quickly ensure all packages in requirements.txt are installed.
# If they already are, it finishes in seconds.
python3 -m pip install -r requirements.txt

echo "Performing system health check..."
# pip check scans EVERY installed library to ensure there are no version conflicts.
python3 -m pip check

if [ $? -ne 0 ]; then
    echo ""
    echo "[!] WARNING: Environment corruption or incompatible libraries detected!"
    echo "[!] Auto-fixing the environment now. This may take a minute..."
    echo ""
    
    # Force a clean reinstall of all strictly pinned libraries in your requirements
    python3 -m pip install --force-reinstall -r requirements.txt
    
    # Verify health check passed after auto-fix
    python3 -m pip check
    
    if [ $? -ne 0 ]; then
        echo "[!] ERROR: Could not auto-resolve all conflicts. Please delete the .venv folder and run setup.sh again."
        exit 1
    else
        echo "[+] Environment successfully repaired!"
    fi
else
    echo "[+] Environment is healthy."
fi

echo ""
echo "Launching application..."
python3 app.py
