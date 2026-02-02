# mistrachat_langchain.py - VERSION FINALE (format structuré + saut de ligne)

import os
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from utils.query_classifier import QueryClassifier
import re

# 1. Variables d'environnement
load_dotenv()
API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    st.error("❌ MISTRAL_API_KEY manquante")
    st.stop()

# 2. Clients
mistral = Mistral(api_key=API_KEY)
CHAT_MODEL = "mistral-large-latest"
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=API_KEY)

# 3. Charger index LangChain
VECTORSTORE_PATH = "data/vector_index_langchain"
if not os.path.exists(VECTORSTORE_PATH):
    st.error("❌ Index manquant. Exécutez : python reindex_langchain.py")
    st.stop()

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 4. QueryClassifier
query_classifier = QueryClassifier()

# 5. Streamlit
st.set_page_config(page_title="Puls Events", page_icon="🗓️")
st.title("🗓️ Assistant Puls Events - Montpellier")
st.caption(f"🤖 {CHAT_MODEL} | LangChain FAISS")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Bonjour ! Puls Events Montpellier prêt ! 🎭🎶"}]

# 6. Fonctions
def parse_event_info(page_content: str):
    """Extrait titre, description, lieu, date, URL du page_content"""
    patterns = {
        'titre': r'Titre:\s*(.+?)\s*\|\s*Adresse:',
        'adresse': r'Adresse:\s*(.+?)\s*\|\s*Date:',
        'date': r'Date:\s*(.+?)\s*\|\s*URL:',
        'url': r'URL:\s*(.+?)\s*\|\s*Description:',
        'description': r'Description:\s*(.+)$'
    }
    
    info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, page_content, re.DOTALL)
        if match:
            info[key] = match.group(1).strip()
    
    return info

def format_events_list(docs: list):
    """Formate les événements en liste structurée"""
    events_list = []
    for doc in docs:
        info = parse_event_info(doc.page_content)
        if info.get('titre'):
            events_list.append(f"""- **{info.get('titre', 'N/A')}**
  - **Lieu** : {info.get('adresse', 'N/A')}
  - **Date** : {info.get('date', 'N/A')}
  
  Pour plus d'informations : {info.get('url', 'N/A')}""")
    
    return "\n\n".join(events_list)

def build_rag_prompt(question: str, docs: list) -> str:
    events_formatted = format_events_list(docs)
    return f"""Expert événements Montpellier.

Utilise UNIQUEMENT ces informations structurées pour répondre.

ÉVÉNEMENTS TROUVÉS :
{events_formatted}

QUESTION: {question}

RÉPONSE (format naturel, liste les événements trouvés avec leurs détails) :"""

def generate_response(prompt: str) -> str:
    try:
        resp = mistral.chat.complete(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur: {e}"

# 7. Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Que cherchez-vous à Montpellier ?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # RAG ou Direct
    needs_rag, _, _ = query_classifier.needs_rag(prompt)
    docs = vectorstore.similarity_search(prompt, k=3) if needs_rag else []

    # Prompt structuré
    rag_prompt = build_rag_prompt(prompt, docs) if needs_rag else prompt

    # Réponse
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("🤔 Recherche...")
        response = generate_response(rag_prompt)
        placeholder.success(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Reset
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "👋 Prêt ! 🎭🎶"}]
        st.rerun()
