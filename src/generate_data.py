"""
Génère les jeux de données volontairement dégradés utilisés dans les ateliers du module
Préparation et Manipulation de Données (M1).

Usage :
    python src/generate_data.py                # fichiers de base dans data/raw/
    python src/generate_data.py --big 5000000  # ajoute un fichier volumineux pour la séance 7

Les défauts injectés (valeurs manquantes, doublons, formats de date hétérogènes, unités
mélangées, valeurs aberrantes, casse incohérente) sont intentionnels : ce sont les objets
d'étude des séances 2 à 6.

Convention : le code est en anglais et suit la PEP 8. Les noms de colonnes des fichiers
produits restent en français, comme le seraient ceux d'une source métier française.
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

CITIES = ["Paris", "Lyon", "Marseille", "Lille", "Bordeaux", "Nantes", "Toulouse", "Strasbourg"]
REGIONS = {
    "Paris": "Ile-de-France", "Lyon": "Auvergne-Rhone-Alpes", "Marseille": "PACA",
    "Lille": "Hauts-de-France", "Bordeaux": "Nouvelle-Aquitaine", "Nantes": "Pays de la Loire",
    "Toulouse": "Occitanie", "Strasbourg": "Grand Est",
}
CATEGORIES = {
    "Informatique": ["Ordinateur portable", "Ecran", "Clavier", "Souris"],
    "Audio": ["Casque", "Enceinte", "Micro"],
    "Mobilier": ["Bureau", "Chaise", "Lampe"],
    "Accessoire": ["Sacoche", "Cable", "Adaptateur"],
}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d %H:%M", "%d %b %Y"]


def messy_case(value, rng):
    """Reproduit la saisie humaine : casse instable et espaces parasites."""
    draw = rng.random()
    if draw < 0.12:
        return value.upper()
    if draw < 0.24:
        return value.lower()
    if draw < 0.32:
        return "  " + value + " "
    return value


def format_date(moment, rng):
    """Renvoie la même date dans l'un des cinq formats rencontrés en production."""
    return moment.strftime(DATE_FORMATS[rng.integers(0, len(DATE_FORMATS))])


def generate_products(rng):
    rows = []
    product_number = 1
    for category, subcategories in CATEGORIES.items():
        for subcategory in subcategories:
            for variant in range(1, 4):
                rows.append({
                    "id_produit": f"P{product_number:04d}",
                    "libelle": f"{subcategory} modele {variant}",
                    "categorie": category,
                    "sous_categorie": subcategory,
                    "cout_achat": float(np.round(rng.uniform(8, 620), 2)),
                })
                product_number += 1

    products = pd.DataFrame(rows)
    missing_index = rng.choice(products.index, size=int(len(products) * 0.03), replace=False)
    products.loc[missing_index, "cout_achat"] = np.nan
    return products


def generate_stores(rng):
    rows = []
    for number, city in enumerate(CITIES, start=1):
        rows.append({
            "id_magasin": f"M{number:02d}",
            "nom_magasin": f"Boutique {city}",
            "ville": city,
            "region": REGIONS[city],
            "surface_m2": int(rng.integers(120, 900)),
        })
    rows.append({
        "id_magasin": "M99", "nom_magasin": "Entrepot central",
        "ville": "Orleans", "region": "Centre-Val de Loire", "surface_m2": 4200,
    })
    return pd.DataFrame(rows)


def generate_customers(rng, n_customers=3000):
    start = datetime(2019, 1, 1)
    rows = []
    for number in range(1, n_customers + 1):
        city = CITIES[rng.integers(0, len(CITIES))]
        signup = start + timedelta(days=int(rng.integers(0, 2000)))
        rows.append({
            "id_client": f"C{number:05d}",
            "date_inscription": signup.strftime("%Y-%m-%d"),
            "segment": ["Particulier", "Professionnel", "Association"][
                int(rng.choice([0, 0, 0, 1, 1, 2]))],
            "ville": messy_case(city, rng),
            "age": int(rng.normal(41, 13)),
        })

    customers = pd.DataFrame(rows)

    # Âges aberrants : sentinelles de saisie et valeurs impossibles
    outlier_index = rng.choice(customers.index, size=45, replace=False)
    customers.loc[outlier_index[:20], "age"] = 999
    customers.loc[outlier_index[20:32], "age"] = 0
    customers.loc[outlier_index[32:], "age"] = -1

    missing_index = rng.choice(customers.index, size=int(len(customers) * 0.06), replace=False)
    customers.loc[missing_index, "age"] = np.nan

    # Doublons métier : même personne ressaisie sous un autre identifiant
    duplicates = customers.sample(60, random_state=SEED).copy()
    duplicates["id_client"] = [f"C9{90000 + i:04d}"[-6:] for i in range(len(duplicates))]
    return pd.concat([customers, duplicates], ignore_index=True)


def generate_sales(rng, products, customers, stores, n_orders=18000):
    start = datetime(2023, 1, 1)
    product_ids = products["id_produit"].tolist()
    customer_ids = customers["id_client"].tolist()
    store_ids = stores["id_magasin"].tolist()
    reference_price = dict(zip(products["id_produit"],
                               products["cout_achat"].fillna(50) * 1.9))

    rows = []
    for number in range(1, n_orders + 1):
        moment = start + timedelta(days=int(rng.integers(0, 730)),
                                   hours=int(rng.integers(8, 21)),
                                   minutes=int(rng.integers(0, 60)))

        # Saisonnalité : pic en novembre et décembre, creux en août
        if moment.month in (11, 12) and rng.random() < 0.35:
            quantity = int(rng.integers(2, 7))
        elif moment.month == 8 and rng.random() < 0.4:
            continue
        else:
            quantity = int(rng.integers(1, 4))

        product_id = product_ids[rng.integers(0, len(product_ids))]

        rows.append({
            "id_commande": f"CMD{number:07d}",
            "date_commande": format_date(moment, rng),
            "id_client": customer_ids[rng.integers(0, len(customer_ids))],
            "id_produit": product_id,
            "id_magasin": store_ids[rng.integers(0, len(store_ids))],
            "quantite": quantity,
            "prix_unitaire": float(np.round(reference_price[product_id]
                                            * rng.uniform(0.92, 1.12), 2)),
            "remise_pct": float(np.round(rng.choice([0, 0, 0, 5, 10, 15, 20]), 1)),
            "canal": messy_case(["web", "boutique", "telephone"][
                int(rng.choice([0, 0, 0, 1, 1, 2]))], rng),
            "statut": ["livre", "livre", "livre", "livre", "annule", "retourne"][
                int(rng.integers(0, 6))],
        })

    sales = pd.DataFrame(rows)

    # --- Injection des défauts ---
    # 1. Prix en texte, virgule décimale et symbole monétaire (7 % des lignes)
    text_price_index = rng.choice(sales.index, size=int(len(sales) * 0.07), replace=False)
    sales["prix_unitaire"] = sales["prix_unitaire"].astype(object)
    sales.loc[text_price_index, "prix_unitaire"] = (
        sales.loc[text_price_index, "prix_unitaire"]
        .apply(lambda price: f"{price:.2f}".replace(".", ",") + " EUR"))

    # 2. Remises manquantes
    missing_index = rng.choice(sales.index, size=int(len(sales) * 0.11), replace=False)
    sales.loc[missing_index, "remise_pct"] = np.nan

    # 3. Quantités aberrantes (erreurs de saisie clavier)
    outlier_index = rng.choice(sales.index, size=90, replace=False)
    sales.loc[outlier_index[:60], "quantite"] = rng.integers(300, 1200, size=60)
    sales.loc[outlier_index[60:], "quantite"] = -rng.integers(1, 5, size=30)

    # 4. Villes de livraison, dont certaines hors périmètre
    sales["ville_livraison"] = [messy_case(CITIES[rng.integers(0, len(CITIES))], rng)
                                for _ in range(len(sales))]
    foreign_index = rng.choice(sales.index, size=120, replace=False)
    sales.loc[foreign_index, "ville_livraison"] = "Geneve"

    # 5. Identifiants produit inconnus (clé cassée)
    broken_key_index = rng.choice(sales.index, size=75, replace=False)
    sales.loc[broken_key_index, "id_produit"] = "P9999"

    # 6. Dates manquantes
    no_date_index = rng.choice(sales.index, size=int(len(sales) * 0.02), replace=False)
    sales.loc[no_date_index, "date_commande"] = np.nan

    # 7. Doublons stricts de lignes
    duplicates = sales.sample(240, random_state=SEED)
    sales = pd.concat([sales, duplicates], ignore_index=True)

    return sales.sample(frac=1, random_state=SEED).reset_index(drop=True)


def generate_sensors(rng):
    """Séries horaires sur 18 mois, avec panne et capteur bloqué."""
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=hour) for hour in range(24 * 540)]
    baselines = {"CAP-A": 12.0, "CAP-B": 14.5, "CAP-C": 11.0}

    rows = []
    for sensor_id, baseline in baselines.items():
        for moment in timestamps:
            seasonal = 9 * np.sin(2 * np.pi * (moment.timetuple().tm_yday - 100) / 365)
            daily = 4 * np.sin(2 * np.pi * (moment.hour - 6) / 24)
            rows.append({
                "horodatage": moment.strftime("%Y-%m-%d %H:%M:%S"),
                "id_capteur": sensor_id,
                "temperature_c": round(float(baseline + seasonal + daily + rng.normal(0, 1.4)), 2),
                "humidite_pct": round(float(np.clip(rng.normal(68 - seasonal, 9), 5, 100)), 1),
                "pm25": round(float(max(0, rng.gamma(2.2, 6))), 1),
            })

    sensors = pd.DataFrame(rows)

    # Panne de CAP-B pendant 11 jours : les lignes n'existent pas
    outage = ((sensors["id_capteur"] == "CAP-B")
              & (sensors["horodatage"] >= "2024-06-03")
              & (sensors["horodatage"] < "2024-06-14"))
    sensors = sensors[~outage].copy()

    # CAP-C bloqué sur une valeur constante pendant 5 jours
    stuck = ((sensors["id_capteur"] == "CAP-C")
             & (sensors["horodatage"] >= "2024-09-10")
             & (sensors["horodatage"] < "2024-09-15"))
    sensors.loc[stuck, "temperature_c"] = 17.4

    # Valeurs manquantes dispersées et sentinelles -999
    missing_index = rng.choice(sensors.index, size=int(len(sensors) * 0.03), replace=False)
    sensors.loc[missing_index, "temperature_c"] = np.nan
    sentinel_index = rng.choice(sensors.index, size=400, replace=False)
    sensors.loc[sentinel_index, "humidite_pct"] = -999

    # Horodatages dupliqués (double envoi du capteur)
    duplicates = sensors.sample(300, random_state=SEED)
    sensors = pd.concat([sensors, duplicates], ignore_index=True)
    return sensors.sort_values(["id_capteur", "horodatage"]).reset_index(drop=True)


def generate_large_file(rng, n_rows, path, chunk_size=500_000):
    """Fichier volumineux pour la comparaison Pandas / PySpark (séance 7)."""
    start = datetime(2022, 1, 1)
    write_header, write_mode, written = True, "w", 0

    while written < n_rows:
        size = min(chunk_size, n_rows - written)
        chunk = pd.DataFrame({
            "id_evenement": np.arange(written, written + size),
            "horodatage": [start + timedelta(seconds=int(offset))
                           for offset in rng.integers(0, 63_072_000, size=size)],
            "id_utilisateur": rng.integers(1, 400_000, size=size),
            "categorie": rng.choice(list(CATEGORIES.keys()), size=size),
            "ville": rng.choice(CITIES, size=size),
            "montant": np.round(rng.gamma(3.0, 22.0, size=size), 2),
            "duree_s": rng.integers(1, 3600, size=size),
        })
        chunk.to_csv(path, mode=write_mode, header=write_header, index=False)
        write_header, write_mode = False, "a"
        written += size
        print(f"  {written} lignes écrites")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--big", type=int, default=0,
                        help="nombre de lignes du fichier volumineux (séance 7)")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    print("Génération des référentiels...")
    products = generate_products(rng)
    stores = generate_stores(rng)
    customers = generate_customers(rng)

    print("Génération des ventes...")
    sales = generate_sales(rng, products, customers, stores)

    print("Génération des mesures de capteurs...")
    sensors = generate_sensors(rng)

    products.to_csv(os.path.join(RAW_DIR, "produits.csv"), index=False)
    stores.to_csv(os.path.join(RAW_DIR, "magasins.csv"), index=False)
    customers.to_csv(os.path.join(RAW_DIR, "clients.csv"), index=False)
    sales.to_csv(os.path.join(RAW_DIR, "ventes_brutes.csv"), index=False)
    sensors.to_csv(os.path.join(RAW_DIR, "capteurs.csv"), index=False)

    # Le même extrait en Excel et en JSON, pour l'atelier 2.1
    sample = sales.head(4000)
    try:
        sample.to_excel(os.path.join(RAW_DIR, "ventes_extrait.xlsx"), index=False)
    except Exception as error:  # openpyxl absent
        print(f"  (export Excel ignoré : {error})")
    sample.to_json(os.path.join(RAW_DIR, "ventes_extrait.json"), orient="records")

    print(f"\nFichiers écrits dans {RAW_DIR} :")
    for name in sorted(os.listdir(RAW_DIR)):
        size_kb = os.path.getsize(os.path.join(RAW_DIR, name)) / 1024
        print(f"  {name:<28} {size_kb:>10.0f} Ko")

    if args.big:
        large_path = os.path.join(RAW_DIR, "evenements_volumineux.csv")
        print(f"\nGénération du fichier volumineux ({args.big} lignes)...")
        generate_large_file(rng, args.big, large_path)
        size_mb = os.path.getsize(large_path) / (1024 ** 2)
        print(f"  {large_path} : {size_mb:.0f} Mo")


if __name__ == "__main__":
    main()