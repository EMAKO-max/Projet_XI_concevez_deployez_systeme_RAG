# reindex_langchain.py - SANS CHUNKS (version LangChain v0.2+)

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document  # ← CORRIGÉ
from pathlib import Path

load_dotenv()
API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    print("ERREUR: MISTRAL_API_KEY manquante dans .env")
    exit(1)

CSV_PATH = "montpellier_2025.csv"
if not os.path.exists(CSV_PATH):
    print(f"ERREUR: {CSV_PATH} non trouve")
    exit(1)

print("Chargement CSV...")
df = pd.read_csv(CSV_PATH)
print(f"{len(df)} evenements charges")

# 1. Documents LangChain DIRECTS (SANS splitter)
docs = []
for _, row in df.iterrows():
    title = str(row.get('title_fr', row.get('title', '')))
    address = str(row.get('location_address', row.get('location_city', 'Montpellier')))
    date = str(row.get('daterange_fr', row.get('firstdate_begin', '')))
    url = str(row.get('canonicalurl', ''))
    desc = str(row.get('description_fr', row.get('description', '')))
    
    # Text COMPLET
    text = f"Titre: {title} | Adresse: {address} | Date: {date} | URL: {url} | Description: {desc}"
    
    if text.strip():
        doc = Document(page_content=text, metadata={'source': 'montpellier_2025.csv'})
        docs.append(doc)

print(f"{len(docs)} documents (text original complet)")

# 2. Embeddings + Index
print("Embeddings Mistral...")
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=API_KEY)

print("Index FAISS...")
vectorstore = FAISS.from_documents(docs, embeddings)

# 3. Sauvegarde
output_dir = Path("data/vector_index_langchain")
output_dir.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(output_dir))
print(f"Index sauve: {output_dir}")

# 4. Test
print("TEST recherche:")
docs_test = vectorstore.similarity_search("fete musique montpellier", k=3)
for i, doc in enumerate(docs_test):
    print(f"{i+1}. {len(doc.page_content)} chars: {doc.page_content[:150]}...")

print("INDEX PRET!")
