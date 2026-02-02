# extraction_montpellier_2025.py
import requests
import pandas as pd
import re
from pathlib import Path

def fetch_montpellier_2025():
    """Récupère uniquement les événements 2025 dont location_city == Montpellier"""
    
    base_url = "https://public.opendatasoft.com/api/records/1.0/search/"
    
    all_events = []
    offset = 0
    limit = 1000

    print("📥 Récupération événements Montpellier 2025...")

    while True:
        params = {
            "dataset": "evenements-publics-openagenda",
            "rows": limit,
            "start": offset,
            "refine.location_city": "Montpellier",
            "refine.firstdate_begin": "2025"
        }

        r = requests.get(base_url, params=params)
        data = r.json()

        events = data.get("records", [])

        # STOP quand plus rien
        if not events:
            break

        all_events.extend(events)
        offset += limit
        print(f"   {len(all_events)} événements Montpellier (2025) récupérés...")

    return all_events


def clean_field(html):
    """Enlève les balises HTML pour description"""
    if not html:
        return ""
    return re.sub(r"<[^>]+>", "", str(html))


def extract_fields(event):
    """Nettoyage + extraction des champs réels du dataset"""
    f = event.get("fields", {})

    return {
        "uid": f.get("uid", ""),
        "title_fr": f.get("title_fr", ""),
        "description_fr": clean_field(f.get("description_fr", "")),
        "location_city": f.get("location_city", ""),
        "location_name": f.get("location_name", ""),
        "location_address": f.get("location_address", ""),
        "firstdate_begin": f.get("firstdate_begin", ""),
        "firstdate_end": f.get("firstdate_end", ""),
        "daterange_fr": f.get("daterange_fr", ""),
        "canonicalurl": f.get("canonicalurl", ""),
    }


if __name__ == "__main__":
    
    events = fetch_montpellier_2025()

    print(f"\n📦 Total événements récupérés : {len(events)}")

    extracted = []
    for e in events:
        row = extract_fields(e)
        
        # Texte RAG
        row["text_for_rag"] = (
            f"{row['title_fr']} | "
            f"{row['description_fr']} | "
            f"{row['location_address']} | "
            f"{row['canonicalurl']} | "
            f"{row['daterange_fr']}"
        )
        
        extracted.append(row)

    df = pd.DataFrame(extracted)

    # Création du dossier
    Path("data/csv").mkdir(parents=True, exist_ok=True)

    output_file = "data/csv/montpellier_2025.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n🎉 CSV FINAL CRÉÉ : {output_file}")
    print(f"📊 Nombre d’événements : {len(df)}\n")
    print(df[["title_fr", "location_city", "firstdate_begin"]].head())
