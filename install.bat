@echo off
echo === Instalando PDF Chatbot CPU ===
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo === Instalacion Completa. Ejecuta: python app.py ===
pause
