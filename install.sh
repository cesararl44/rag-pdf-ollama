#!/bin/bash
echo "=== Instalando PDF Chatbot CPU ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "=== Instalación Completa. Ejecuta: python app.py ==="
