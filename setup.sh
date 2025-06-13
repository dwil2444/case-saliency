#!/bin/bash
if [ ! -d "venv" ]; then
    mkdir venv
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/vaynexie/CWOX.git
cd CWOX
pip install -e .
