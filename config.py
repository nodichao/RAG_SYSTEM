
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# 1. Chargement des documents PDF
documents = []

pdf_files = Path("./documents").glob("*.pdf")

for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    loaded_documents = loader.load()
    documents.extend(loaded_documents)

if not documents:
    raise ValueError("Aucun document PDF trouvé dans ./documents")

print(f"Nombre de pages chargées : {len(documents)}")


# 2. Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Nombre de chunks générés : {len(chunks)}")


# 3. Initialisation des embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Création de la base vectorielle Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(
    f"Indexation terminée ! "
    f"{len(chunks)} chunks enregistrés dans ./chroma_db."
)
