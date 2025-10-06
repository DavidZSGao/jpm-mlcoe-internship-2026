#!/bin/bash
set -e

echo "Installing Python dependencies for JPM MLCOE project..."
python -m pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "Found requirements.txt - installing..."
  python -m pip install -r requirements.txt
else
  echo "No requirements.txt found - installing core dependencies..."
  python -m pip install tensorflow tensorflow-probability pandas numpy matplotlib pytest
fi

echo "Dependencies installed successfully!"
