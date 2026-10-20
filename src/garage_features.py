"""Variables dérivées à valeur métier pour le projet garage (séance 4)."""
import numpy as np
import pandas as pd

NOT_BILLED_STATUSES = ['reprise_gratuite', 'en_litige']
KM_BINS = [0, 50_000, 100_000, 150_000, 200_000, np.inf]
KM_LABELS = ['< 50k', '50-100k', '100-150k', '150-200k', '> 200k']


def add_derived_features(interventions):
    """Ajoute 5 variables dérivées. L'entrée doit contenir les colonnes des référentiels
    prestations (cout_reference_eur) et clients (date_creation)."""
    df = interventions.copy()

    # 1. L'intervention est-elle rentable ? montant facturé - coût de référence
    df['marge_estimee_eur'] = df['montant_facture_eur'] - df['cout_reference_eur']

    # 2. Quelle productivité ? chiffre d'affaires par heure de main-d'œuvre
    #    (durée nulle ou manquante -> pas de calcul, jamais de division par zéro)
    valid_duration = df['duree_heures'].where(df['duree_heures'] > 0)
    df['ca_par_heure_eur'] = df['montant_facture_eur'] / valid_duration

    # 3. L'intervention a-t-elle échoué commercialement ? (reprise gratuite ou litige)
    df['non_aboutie'] = df['statut'].isin(NOT_BILLED_STATUSES)

    # 4. Depuis combien de temps le client est-il connu ? (fidélité)
    df['anciennete_client_jours'] = (df['date_evenement'].dt.normalize()
                                     - df['date_creation']).dt.days
    # Incohérence source : intervention datée AVANT la création du client
    df['incoherence_date_client'] = df['anciennete_client_jours'] < 0

    # 5. Usure du véhicule : tranche de kilométrage (lisible en BI)
    df['tranche_km'] = pd.cut(df['km_vehicule'], bins=KM_BINS, labels=KM_LABELS,
                              right=False).astype('string').fillna('inconnu')
    return df
