import os
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
    print("\n¡TODOS LOS TESTS PASARON EXITOSAMENTE!")

if __name__ == "__main__":
    test_rag_pipeline()
