import os
import gc
import warnings
import gradio as gr

# 1. Definimos un tema base sencillo con paleta de color naranja
custom_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="slate"
)

# 2. Inyectamos CSS para forzar el color neón en los botones
custom_css = """
/* Botón principal (Procesar Documento) */
.primary-btn, button.primary {
    background: linear-gradient(135deg, #ff6b00 0%, #ff8800 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    box-shadow: 0 0 10px rgba(255, 107, 0, 0.4) !important;
}

/* Hover sobre el botón */
.primary-btn:hover, button.primary:hover {
    background: linear-gradient(135deg, #ff8800 0%, #ffa500 100%) !important;
    box-shadow: 0 0 15px rgba(255, 136, 0, 0.7) !important;
}
"""

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

import config

embeddings = None
vector_store = None
llm = None

def init_models():
    global embeddings, llm
    print("1/2 -> Cargando modelo de Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print("2/2 -> Conectando con Ollama (Llama 3.2)...")
    llm = ChatOllama(
        model=config.LLM_MODEL_NAME,  # Usa la variable centralizada de config.py
        temperature=0.1
    )
    print("Modelos listos.")

def process_pdf(file, progress=gr.Progress()):
    global vector_store
    if file is None:
        return "[!] Por favor sube un archivo PDF válido."
    
    try:
        progress(0.1, desc="Extrayendo texto del PDF...")
        loader = PDFPlumberLoader(file.name)
        documents = loader.load()
        
        if not documents:
            return "[X] No se pudo extraer texto del PDF."

        progress(0.4, desc="Dividiendo texto en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)

        progress(0.7, desc="Indexando en la base de datos vectorial...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        gc.collect()
        progress(1.0, desc="Completado!")
        return f"[OK] Documento procesado correctamente ({len(chunks)} fragmentos generados)."
    
    except Exception as e:
        return f"[X] Error procesando el PDF: {str(e)}"

def respond(message, history):
    global vector_store, llm
    
    if vector_store is None:
        return "[!] Primero debes cargar un archivo PDF y presionar 'Procesar Documento'."
    
    try:
        # Recuperar 5 fragmentos relevantes
        docs = vector_store.similarity_search(message, k=5)
        
        if not docs:
            return "No se encontró información relevante en el documento."

        context = "\n\n---\n\n".join([doc.page_content.strip() for doc in docs])
        
        # PROMPT EQUILIBRADO
        prompt = (
            f"Eres un asistente virtual experto que analiza reglamentos y documentos en español.\n"
            f"Usa los siguientes fragmentos extraídos del documento para responder la pregunta o tema del usuario.\n\n"
            f"REGLAS:\n"
            f"- Responde en español de forma clara, directa y fundamentada.\n"
            f"- Basa tu respuesta en el contexto proporcionado.\n"
            f"- Si el contexto no contiene información sobre el tema, di amablemente: 'El documento no menciona información al respecto'.\n\n"
            f"CONTEXTO DEL DOCUMENTO:\n{context}\n\n"
            f"PREGUNTA O TEMA DEL USUARIO: {message}\n\n"
            f"RESPUESTA:"
        )

        response = llm.invoke(prompt)
        gc.collect()
        
        return response.content

    except Exception as e:
        return f"[X] Error generando la respuesta: {str(e)}"
def clear_session():
    global vector_store
    vector_store = None
    gc.collect()
    return None, [], "Estado reiniciado."

if __name__ == "__main__":
    init_models()

   


with gr.Blocks(title="Chatbot RAG PDF (Local con Ollama)", theme=custom_theme, css=custom_css) as demo:
        gr.Markdown("# 📄 Chatbot RAG para PDFs")
        gr.Markdown("Suba un PDF, presione **Procesar Documento** y realice sus consultas.")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Cargar PDF", file_types=[".pdf"])
                btn_process = gr.Button("Procesar Documento", variant="primary")
                status_output = gr.Textbox(label="Estado", interactive=False)
                btn_clear = gr.Button("Limpiar Sesión")

            with gr.Column(scale=2):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    textbox=gr.Textbox(placeholder="Escribe tu pregunta sobre el documento...", scale=7),
                    title=None
                )

        btn_process.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_output])
        btn_clear.click(fn=clear_session, inputs=[], outputs=[pdf_input, chatbot.chatbot, status_output])

        demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        inbrowser=True
    )