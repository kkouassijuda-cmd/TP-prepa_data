"""Rapport d'exploration réutilisable (séance 2)."""
import pandas as pd


def profile_dataframe(df, name='jeu de données', max_cardinality=25):
    """Affiche un rapport d'exploration standard et renvoie un résumé par colonne.

    Produit, dans cet ordre :
      1. le nom, les dimensions et l'empreinte mémoire ;
      2. le nombre de lignes strictement dupliquées ;
      3. le taux de valeurs manquantes par colonne ;
      4. pour chaque colonne : type, nombre de valeurs distinctes, exemples ;
      5. les modalités des colonnes texte de faible cardinalité (révèle les faux niveaux).
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f'=== {name} : {df.shape[0]} lignes × {df.shape[1]} colonnes '
          f'({memory_mb:.1f} Mo)')
    print(f'Doublons stricts : {df.duplicated().sum()}')

    summary = pd.DataFrame({
        'type': df.dtypes.astype(str),
        'manquants': df.isna().sum(),
        'pct_manquants': (df.isna().mean() * 100).round(2),
        'distincts': df.nunique(),
        'exemples': [df[column].dropna().unique()[:3].tolist() for column in df.columns],
    })
    print(summary.to_string())

    text_columns = df.select_dtypes(include=['object', 'string']).columns
    for column in text_columns:
        if df[column].nunique() <= max_cardinality:
            print(f'\n--- modalités de {column} ---')
            print(df[column].value_counts(dropna=False).to_string())
    print()
    return summary
