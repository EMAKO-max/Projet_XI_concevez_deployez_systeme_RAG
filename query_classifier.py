"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
pour l'assistant Puls-Events Montpellier
"""

import re
import logging
import os
from typing import Tuple
from dotenv import load_dotenv
from mistralai import Mistral  # SDK v1

# --- Chargement variables d'environnement ---
load_dotenv()  # lit le fichier .env
API_KEY = os.getenv("MISTRAL_API_KEY")
COMMUNE_NAME = os.getenv("COMMUNE_NAME", "Montpellier")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-large-latest")

if not API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY non défini dans le fichier .env")

class QueryClassifier:
    """Classe pour classifier les requêtes et déterminer si elles nécessitent RAG"""
    def __init__(self):
        self.mistral_client = Mistral(api_key=API_KEY)
        
        # Mots-clés liés à la ville et aux événements
        self.commune_keywords = [
            COMMUNE_NAME.lower(),
            "concert", "spectacle", "événement", "manifestation",
            "festival", "fête", "musique", "opéra", "exposition",
            "place", "rue", "parc", "gratuit", "ouvert au public"
        ]
        
        # Questions générales qui ne nécessitent pas de RAG
        self.general_patterns = [
            r"^(bonjour|salut|hello|coucou|hey|bonsoir)[\s\.,!]*$",
            r"^(merci|thanks|thank you|je te remercie)[\s\.,!]*$",
            r"^(comment ça va|ça va|comment vas-tu|comment allez-vous)[\s\.,!?]*$",
            r"^(au revoir|bye|à bientôt|à plus tard)[\s\.,!]*$",
        ]

    def needs_rag(self, query: str) -> Tuple[bool, float, str]:
        query_lower = query.lower().strip()
        
        # 1. Questions générales
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return False, 0.95, "Question générale ou salutation"
        
        # 2. Mots-clés liés à la ville / événements
        keywords_found = [kw for kw in self.commune_keywords if kw in query_lower]
        if keywords_found:
            return True, 0.9, f"Contient mots-clés pertinents: {', '.join(keywords_found)}"
        
        # 3. Cas ambigus -> LLM
        return self._classify_with_llm(query)
    
    def _classify_with_llm(self, query: str) -> Tuple[bool, float, str]:
        try:
            system_prompt = f"""
            Vous êtes un classificateur de requêtes pour un assistant sur les événements à {COMMUNE_NAME}.
            Répondez uniquement par "RAG" ou "DIRECT" suivi d'une brève explication.
            - "RAG": question spécifique à {COMMUNE_NAME} (concerts, festivals, spectacles, lieux, horaires, etc.)
            - "DIRECT": question générale ou salutation.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.mistral_client.chat.complete(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            logging.info(f"Classification LLM pour '{query}': {result}")
            
            if result.startswith("RAG"):
                return True, 0.85, result.replace("RAG -", "").strip()
            elif result.startswith("DIRECT"):
                return False, 0.85, result.replace("DIRECT -", "").strip()
            else:
                return True, 0.6, "Classification ambiguë, utilisation de RAG par précaution"
        
        except Exception as e:
            logging.error(f"Erreur classification LLM: {e}")
            return True, 0.5, f"Erreur LLM: {str(e)}"


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    qc = QueryClassifier()
    test_queries = [
        "Quels concerts gratuits auront lieu à Montpellier en juin 2025 ?",
        "Bonjour",
        "Y a-t-il des visites libres à l’Opéra Comédie cette année ?",
        "Qu'est-ce que l'intelligence artificielle ?",
    ]
    
    for q in test_queries:
        rag, conf, reason = qc.needs_rag(q)
        print(f"Question: {q}\nRAG: {rag}, Confiance: {conf}, Raison: {reason}\n")
