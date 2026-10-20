"""Jointures sûres et modèle en étoile (séance 6)."""
import pandas as pd


def safe_merge(left, right, **kwargs):
    """merge() qui refuse l'explosion de lignes et signale les clés non appariées."""
    how = kwargs.get('how', 'inner')
    merged = left.merge(right, indicator='_merge_status', **kwargs)

    unmatched = int((merged['_merge_status'] == 'left_only').sum())
    rate = unmatched / len(left) if len(left) else 0.0
    print(f'safe_merge : {len(left)} -> {len(merged)} lignes | '
          f'clés non appariées : {unmatched} ({rate:.2%})')

    if how == 'left' and len(merged) != len(left):
        raise ValueError(f'left join : {len(left)} lignes attendues, {len(merged)} obtenues '
                         '(clé non unique dans la table de droite ?)')
    return merged.drop(columns='_merge_status')


def add_unknown_members(dimension, fact_keys, key, defaults):
    """Ajoute à une dimension une ligne « inconnu » pour chaque clé de faits absente.

    Pratique standard en BI : les faits orphelins restent comptés, rangés sous
    'inconnu', au lieu de disparaître des rapports.
    """
    missing = sorted(set(fact_keys.dropna()) - set(dimension[key]))
    if not missing:
        return dimension, missing
    unknown = pd.DataFrame({key: missing, **{col: value for col, value in defaults.items()}})
    return pd.concat([dimension, unknown], ignore_index=True), missing


def build_date_dimension(dates):
    """Dimension temps continue (y compris les jours sans activité)."""
    full_range = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq='D')
    dim_date = pd.DataFrame({'date': full_range})
    dim_date['cle_date'] = dim_date['date'].dt.strftime('%Y%m%d').astype(int)
    dim_date['annee'] = dim_date['date'].dt.year
    dim_date['mois'] = dim_date['date'].dt.month
    dim_date['nom_mois'] = dim_date['date'].dt.month_name()
    dim_date['trimestre'] = dim_date['date'].dt.quarter
    dim_date['semaine_iso'] = dim_date['date'].dt.isocalendar().week.astype(int)
    dim_date['jour_semaine'] = dim_date['date'].dt.dayofweek
    dim_date['nom_jour'] = dim_date['date'].dt.day_name()
    dim_date['est_weekend'] = dim_date['jour_semaine'] >= 5
    return dim_date


def check_referential_integrity(fact_table, dimensions):
    """dimensions : dict {nom_colonne_cle: dataframe_dimension}.

    Affiche, pour chaque dimension, le nombre de clés orphelines et vérifie l'unicité
    de la clé primaire. Lève une ValueError au moindre problème.
    """
    rows = []
    for key, dimension in dimensions.items():
        orphan_mask = ~fact_table[key].isin(dimension[key])
        rows.append({'cle': key,
                     'lignes_faits': len(fact_table),
                     'cles_orphelines': int(orphan_mask.sum()),
                     'exemples': fact_table.loc[orphan_mask, key].unique()[:5].tolist(),
                     'cle_dimension_unique': bool(dimension[key].is_unique)})
    report = pd.DataFrame(rows).set_index('cle')
    print(report.to_string())

    problems = report[(report['cles_orphelines'] > 0) | ~report['cle_dimension_unique']]
    if not problems.empty:
        raise ValueError('Intégrité référentielle violée pour : ' + ', '.join(problems.index))
    print('OK — intégrité référentielle respectée')
    return report
