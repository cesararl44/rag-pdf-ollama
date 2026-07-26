import os
import torch

TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "chroma_db")


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "llama3.2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150