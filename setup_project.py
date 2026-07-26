import os

# Estructura de archivos y sus contenidos
files = {
    "config.py": '''import os
import torch

TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base" 

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
''',

    "requirements.txt": '''gradio==4.44.0
langchain==0.3.1
langchain-community==0.3.1
chromadb==0.5.5
sentence-transformers==2.7.0
pdfplumber==0.11.4
transformers==4.44.2
accelerate==0.34.2
bitsandbytes==0.43.3
torch==2.4.1+cpu
--extra-index-url https://download.pytorch.org/whl/cpu
''',

    "app.py": '''import os
import gc
import torch
import gradio as gr
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import config

embeddings = None
vector_store = None
llm_pipeline = None

def init_models():
    global embeddings, llm_pipeline
    print("Cargando modelo de Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print("Cargando LLM en CPU...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.LLM_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False
    )
    print("Modelos cargados exitosamente.")

def process_pdf(file, progress=gr.Progress()):
    global vector_store
    if file is None:
        return "⚠️ Por favor sube un archivo PDF válido."
    
    try:
        progress(0.1, desc="Extrayendo texto del PDF...")
        loader = PDFPlumberLoader(file.name)
        documents = loader.load()
        
        if not documents:
            return "❌ No se pudo extraer texto del PDF."

        progress(0.4, desc="Dividiendo texto en fragmentos (chunks)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)

        progress(0.7, desc="Generando Embeddings e indexando...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config.VECTOR_STORE_DIR
        )
        
        gc.collect()
        progress(1.0, desc="¡Completado!")
        return f"✅ Documento procesado correctamente ({len(chunks)} fragmentos generados)."
    
    except Exception as e:
        return f"❌ Error procesando el PDF: {str(e)}"

def respond(message, history):
    global vector_store, llm_pipeline
    
    if vector_store is None:
        return "⚠️ Primero debes cargar un archivo PDF antes de realizar preguntas."
    
    try:
        docs = vector_store.similarity_search(message, k=config.TOP_K_RESULTS)
        
        if not docs:
            return "No se encontró información relevante en el documento."

        context = "\\n\\n".join([doc.page_content for doc in docs])
        
        prompt = (
            f"Basándote strictly en el siguiente contexto, responde de manera clara y directa.\\n\\n"
            f"Contexto:\\n{context}\\n\\n"
            f"Pregunta: {message}\\n"
            f"Respuesta:"
        )

        with torch.no_grad():
            response = llm_pipeline(prompt)[0]['generated_text']
            
        gc.collect()
        return response

    except Exception as e:
        return f"❌ Error generando la respuesta: {str(e)}"

def clear_session():
    global vector_store
    vector_store = None
    gc.collect()
    return None, [], "Estado reiniciado."

init_models()

with gr.Blocks(title="Chatbot RAG PDF (CPU)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Chatbot RAG para PDFs (100% CPU / Local)")
    gr.Markdown("Suba un PDF e interactúe de forma totalmente privada sin consumo de APIs de pago.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Cargar PDF", file_types=[".pdf"])
            btn_process = gr.Button("Procesar Documento", variant="primary")
            status_output = gr.Textbox(label="Estado", interactive=False)
            btn_clear = gr.Button("Limpiar Sesión")

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=respond,
                textbox=gr.Textbox(placeholder="Haz una pregunta sobre el documento...", scale=7),
                title=None
            )

    btn_process.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_output])
    btn_clear.click(fn=clear_session, inputs=[], outputs=[pdf_input, chatbot.chatbot, status_output])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
''',

    "Dockerfile": '''FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('google/flan-t5-base'); AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
''',

    "docker-compose.yml": '''version: '3.8'

services:
  pdf-chatbot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: pdf_chatbot_cpu
    ports:
      - "7860:7860"
    deploy:
      resources:
        limits:
          memory: 10g
        reservations:
          memory: 4g
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
''',

    ".env.example": '''HOST=0.0.0.0
PORT=7860
TORCH_NUM_THREADS=4
''',

    "install.sh": '''#!/bin/bash
echo "=== Instalando PDF Chatbot CPU ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "=== Instalación Completa. Ejecuta: python app.py ==="
''',

    "install.bat": '''@echo off
echo === Instalando PDF Chatbot CPU ===
python -m venv venv
call venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo === Instalacion Completa. Ejecuta: python app.py ===
pause
''',

    "tests/test_app.py": '''import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

def test_rag_pipeline():
    print("Iniciando test de extracción y embeddings...")
    sample_pdf_path = os.path.join(os.path.dirname(__file__), 'sample.pdf')
    
    if not os.path.exists(sample_pdf_path):
        print("⚠️ Para correr la prueba, coloca un archivo 'sample.pdf' dentro de la carpeta 'tests/'")
        return

    loader = PDFPlumberLoader(sample_pdf_path)
    docs = loader.load()
    assert len(docs) > 0, "No se extrajo texto del PDF."
    print("✓ Carga de PDF exitosa.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 0, "El chunking falló."
    print("✓ Fragmentación exitosa.")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vector = embeddings.embed_query("Test de consulta")
    assert len(vector) > 0, "Generación de embeddings fallida."
    print("✓ Embeddings vectoriales generados correctamente.")
    print("\\n¡TODOS LOS TESTS PASARON EXITOSAMENTE!")

if __name__ == "__main__":
    test_rag_pipeline()
'''
}

def create_structure():
    print("🚀 Creando estructura del proyecto...")
    # Crear carpetas necesarias
    os.makedirs("tests", exist_ok=True)

    # Crear cada archivo con su contenido
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [+] Creado: {path}")

    # Dar permisos de ejecución a scripts bash si está en linux/mac
    if os.name != 'nt':
        os.chmod("install.sh", 0o755)

    print("\n✅ ¡Todos los archivos han sido generados exitosamente!")

if __name__ == "__main__":
    create_structure()