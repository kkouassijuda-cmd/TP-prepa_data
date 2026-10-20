"""Nettoyage des données du projet garage (groupe 04).

Chaque fonction :
  - copie son entrée (la source n'est jamais modifiée) ;
  - renvoie (df_propre, journal), le journal comptant ce qui a été touché ;
  - est idempotente : la réappliquer à son propre résultat ne change plus rien.
"""
import numpy as np
import pandas as pd

# Colonnes texte DESCRIPTIVES à normaliser. Les identifiants ne sont jamais touchés :
# les passer en minuscules casserait les jointures (cf. séance 3).
INTERVENTION_TEXT_COLUMNS = ['origine', 'statut']
CLIENT_TEXT_COLUMNS = ['segment', 'ville']

KM_SENTINEL = 999_999            # valeur « non renseigné » du système source
MAX_PLAUSIBLE_DURATION_H = 9     # au-delà : improbable (42 lignes à 24 h), signalé
MIN_ADULT_AGE, MAX_AGE = 18, 100  # client d'un garage = conducteur majeur


def normalize_text(series):
    """Casse, espaces et accents : trois sources de faux niveaux."""
    text = series.astype('string').str.strip().str.lower()
    text = text.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('utf-8')
    return text.str.replace(r'\s+', ' ', regex=True)


def parse_amount(series):
    """'1114,64 EUR' -> 1114.64 ; 'nan EUR' -> NaN (valeur manquante déguisée)."""
    text = (series.astype('string')
            .str.replace('EUR', '', regex=False)
            .str.replace(',', '.', regex=False)
            .str.strip())
    return pd.to_numeric(text, errors='coerce')


def parse_dates(series, min_success_rate=0.95):
    """Convertit des dates hétérogènes et renvoie (dates, rapport).

    Lève une ValueError si trop de dates renseignées échouent à la conversion :
    errors='coerce' ne doit jamais détruire des dates en silence.
    """
    if pd.api.types.is_datetime64_any_dtype(series):        # idempotence
        converted = series
    else:
        converted = pd.to_datetime(series, format='mixed', dayfirst=True, errors='coerce')
    n_already_missing = int(series.isna().sum())
    n_to_convert = len(series) - n_already_missing
    n_failed = int(converted.isna().sum()) - n_already_missing
    success_rate = (n_to_convert - n_failed) / n_to_convert if n_to_convert else 0.0
    report = {'n_total': len(series), 'n_already_missing': n_already_missing,
              'n_failed': n_failed, 'success_rate': round(success_rate, 4),
              'min_date': converted.min(), 'max_date': converted.max()}
    if success_rate < min_success_rate:
        raise ValueError(f'taux de conversion des dates {success_rate:.1%} '
                         f'< seuil {min_success_rate:.0%}')
    return converted, report


def clean_interventions(df, verbose=True):
    """Nettoie la table de faits des interventions. Renvoie (df, journal)."""
    log = {'lignes_initiales': len(df)}
    df = df.copy()

    # 1. Doublons stricts : même intervention exportée deux fois
    before = len(df)
    df = df.drop_duplicates()
    log['doublons_stricts_supprimes'] = before - len(df)

    # 2. Montant : texte '1114,64 EUR' -> nombre ; 'nan EUR' -> NaN
    if not pd.api.types.is_numeric_dtype(df['montant_facture_eur']):
        raw = df['montant_facture_eur']
        log['montants_au_format_texte'] = int(raw.astype('string')
                                              .str.contains('EUR', na=False).sum())
        df['montant_facture_eur'] = parse_amount(raw)
    # Montant manquant : on NE l'impute PAS (ce serait inventer du chiffre d'affaires),
    # on le signale par un indicateur.
    df['montant_manquant'] = df['montant_facture_eur'].isna()
    log['montants_manquants'] = int(df['montant_manquant'].sum())

    # 3. Colonnes texte descriptives
    for column in INTERVENTION_TEXT_COLUMNS:
        df[column] = normalize_text(df[column])

    # 4. Dates : 5 formats ; une intervention sans date est inexploitable dans le temps
    df['date_evenement'], date_report = parse_dates(df['date_evenement'])
    log['dates_echec_conversion'] = date_report['n_failed']
    before = len(df)
    df = df.dropna(subset=['date_evenement'])
    log['lignes_sans_date_supprimees'] = before - len(df)

    # 5. Kilométrage : 999 999 = « non renseigné » (sentinelle) -> NaN
    df['km_vehicule'] = df['km_vehicule'].astype('float')
    sentinel = df['km_vehicule'] == KM_SENTINEL
    log['km_sentinelle_999999'] = int(sentinel.sum())
    df.loc[sentinel, 'km_vehicule'] = np.nan

    # 6. Durée : négative = IMPOSSIBLE -> NaN (ligne conservée, son montant reste valable)
    #            24 h = IMPROBABLE (la norme est 1 à 9 h) -> conservée mais signalée
    df['duree_heures'] = df['duree_heures'].astype('float')
    negative = df['duree_heures'] < 0
    log['durees_negatives'] = int(negative.sum())
    df.loc[negative, 'duree_heures'] = np.nan
    df['duree_atypique'] = df['duree_heures'] > MAX_PLAUSIBLE_DURATION_H
    log['durees_atypiques_signalees'] = int(df['duree_atypique'].sum())

    log['lignes_finales'] = len(df)
    if verbose:
        for key, value in log.items():
            print(f'  {key:<32} {value}')
    return df.reset_index(drop=True), log


def clean_clients(df, verbose=True):
    """Nettoie le référentiel clients. Renvoie (df, journal)."""
    log = {'lignes_initiales': len(df)}
    df = df.copy()

    for column in CLIENT_TEXT_COLUMNS:
        df[column] = normalize_text(df[column])
    # Ville affichée avec une majuscule initiale pour les rapports
    df['ville'] = df['ville'].str.title()

    df['date_creation'] = pd.to_datetime(df['date_creation'], errors='coerce')

    # Âge : < 18 ou > 100 est impossible pour un client conducteur (-7, 999...) -> NaN
    df['age'] = df['age'].astype('float')
    invalid_age = df['age'].notna() & ~df['age'].between(MIN_ADULT_AGE, MAX_AGE)
    log['ages_invalides'] = int(invalid_age.sum())
    df.loc[invalid_age, 'age'] = np.nan

    # Puis imputation par la médiane du SEGMENT, avec indicateur (MAR : l'âge dépend
    # du segment, la médiane résiste aux extrêmes). Idempotent : l'indicateur est
    # calculé avant l'imputation et conservé s'il existe déjà.
    if 'age_impute' not in df.columns:
        df['age_impute'] = df['age'].isna()
    log['ages_imputes'] = int(df['age'].isna().sum())
    df['age'] = df['age'].fillna(df.groupby('segment')['age'].transform('median'))

    log['lignes_finales'] = len(df)
    if verbose:
        for key, value in log.items():
            print(f'  {key:<32} {value}')
    return df, log


def clean_prestations(df, verbose=True):
    """Nettoie le référentiel prestations. Renvoie (df, journal)."""
    log = {'lignes_initiales': len(df)}
    df = df.copy()
    df['cout_reference_eur'] = df['cout_reference_eur'].astype('float')
    # Coût manquant : médiane de la sous-famille (prestations comparables), indicateur
    if 'cout_impute' not in df.columns:
        df['cout_impute'] = df['cout_reference_eur'].isna()
    log['couts_imputes'] = int(df['cout_reference_eur'].isna().sum())
    for group in ['sous_famille', 'famille']:          # repli sur la famille si besoin
        df['cout_reference_eur'] = df['cout_reference_eur'].fillna(
            df.groupby(group)['cout_reference_eur'].transform('median'))
    log['lignes_finales'] = len(df)
    if verbose:
        for key, value in log.items():
            print(f'  {key:<32} {value}')
    return df, log
