import pandas as pd
import pytest
from Exxtraction import extract_fields

class TestMontpellierEvents:

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = pd.DataFrame([
            {"uid": "1", "title_fr": "Concert A", "description_fr": "Super concert",
             "location_city": "Montpellier", "firstdate_begin": "2025-05-10"},
            {"uid": "2", "title_fr": "Festival B", "description_fr": "<p>Festival incroyable</p>",
             "location_city": "Montpellier", "firstdate_begin": "2025-07-20"},
            {"uid": "3", "title_fr": "Exposition C", "description_fr": "Expo d'art",
             "location_city": "Paris", "firstdate_begin": "2025-09-15"},
        ])
        # Filtrage comme fetch_montpellier_2025
        self.df_filtered = self.data[
            (self.data["location_city"] == "Montpellier") &
            (self.data["firstdate_begin"].str.contains("2025"))
        ]
        self.extracted = [extract_fields({"fields": row.to_dict()}) for _, row in self.df_filtered.iterrows()]
        self.df_final = pd.DataFrame(self.extracted)
        self.df_final["text_for_rag"] = self.df_final.apply(
            lambda x: f"{x['title_fr']} | {x['description_fr']} | {x['location_address']} | "
                      f"{x['canonicalurl']} | {x['daterange_fr']}", axis=1
        )

    def test_filtered_count(self):
        assert len(self.df_filtered) == 2

    def test_all_montpellier(self):
        assert all(self.df_filtered["location_city"] == "Montpellier")

    def test_extract_fields_keys(self):
        for e in self.extracted:
            assert "uid" in e
            assert "title_fr" in e
            assert "description_fr" in e

    def test_text_for_rag(self):
        for text in self.df_final["text_for_rag"]:
            assert "|" in text
