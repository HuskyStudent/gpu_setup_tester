#!/bin/bash

echo "🔧 Running GPU availability tests..."

echo -e "\n0. Running GPUavail.py..."
python GPUavail.py

echo -e "\n1. Running GPUavail2.py..."
python GPUavail2.py

echo -e "\n2. Running HFavail.py..."
python HFavail.py

echo -e "\n3. Running HFavail.py..."
python HFavail2.py



echo -e "DONE"
