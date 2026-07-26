# 📘 Chatbot RAG para PDFs (100% Local y Privado)

Agente de **Retrieval-Augmented Generation (RAG)** para conversar con documentos PDF en español de forma totalmente local, privada y sin costo por APIs. Desarrollado con **LangChain, Ollama (Llama 3.2), Gradio y ChromaDB**.

---

## 🚀 Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
Crear entorno virtual:

Bash
python -m venv venv
Activar entorno virtual:

Windows (Git Bash): source venv/Scripts/activate

Windows (CMD): venv\Scripts\activate

Linux / Mac: source venv/bin/activate

Instalar dependencias:

Bash
pip install -r requirements.txt
Instalar y preparar Ollama:

Descarga e instala Ollama desde ollama.com.

Descarga el modelo ejecutando en tu terminal (CMD o PowerShell):

Bash
ollama run llama3.2
🛠️ Uso
Activa tu entorno virtual (venv).

Ejecuta la aplicación:

Bash
python app.py
Abre tu navegador en http://127.0.0.1:7860.

Carga tu archivo PDF en el panel lateral.

Presiona "Procesar Documento" para indexar la información.

¡Comienza a realizar preguntas sobre tu documento desde la interfaz web!

🎨 Características y Mejoras de Interfaz
Ejecución 100% Local: No requiere credenciales ni APIs de pago; tus datos no salen de tu computadora.

Soporte PDF Flexible: Extracción y procesamiento automatizado con PDFPlumberLoader.

Búsqueda Multilingüe: Optimizado con embeddings en español para consultas precisas sobre reglamentos y documentos técnicos.

Tema Personalizado: Interfaz responsiva adaptada con Gradio y detalles visuales en tonos vibrantes.

Gestión de Memoria y Sesión: Botón dedicado para reiniciar la base vectorial y liberar recursos (gc.collect()).

Prompting Especializado: Diseñado para responder únicamente con el contexto extraído del documento cargado, evitando alucinaciones.

📚 Tecnologías Utilizadas
Lenguaje: Python 3.10+

Orquestador LLM: LangChain

Modelo de Lenguaje (LLM): Llama 3.2 (vía Ollama)

Embeddings: HuggingFace Embeddings (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

Base de Datos Vectorial: ChromaDB

Interfaz Web: Gradio