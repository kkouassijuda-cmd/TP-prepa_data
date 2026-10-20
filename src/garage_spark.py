"""Composant PySpark du projet garage (séance 7) : nettoyage des interventions."""
import os

from pyspark.sql import functions as F
from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType,
                               StructField, StructType)

# Sous Windows, Spark ne démarre pas si le chemin de pyspark contient un accent
# ('donnée') : on passe par une jonction sans accent si elle existe.
WINDOWS_SPARK_HOME = r'C:\Users\Public\pyspark_home'

# Montant et date lus en TEXTE : les déclarer Double / Timestamp transformerait
# silencieusement '1114,64 EUR' et 4 formats de dates sur 5 en null.
INTERVENTIONS_SCHEMA = StructType([
    StructField('id_intervention', StringType(), True),
    StructField('date_evenement', StringType(), True),
    StructField('id_client', StringType(), True),
    StructField('id_prestation', StringType(), True),
    StructField('id_atelier', StringType(), True),
    StructField('duree_heures', IntegerType(), True),
    StructField('km_vehicule', LongType(), True),
    StructField('montant_facture_eur', StringType(), True),
    StructField('origine', StringType(), True),
    StructField('statut', StringType(), True),
])

DATE_FORMATS = ['yyyy/MM/dd HH:mm', 'yyyy-MM-dd', 'dd-MM-yyyy', 'dd/MM/yyyy', 'dd MMM yyyy']
KM_SENTINEL = 999_999
MAX_PLAUSIBLE_DURATION_H = 9


def get_spark(app_name='garage', memory='2g'):
    """Session Spark locale, réglée pour une machine unique."""
    if os.name == 'nt' and os.path.exists(WINDOWS_SPARK_HOME):
        os.environ['SPARK_HOME'] = WINDOWS_SPARK_HOME
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName(app_name)
             .master('local[*]')
             .config('spark.driver.memory', memory)
             .config('spark.sql.shuffle.partitions', '8')   # 200 par défaut : trop en local
             .config('spark.ui.showConsoleProgress', 'false')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    return spark


def clean_interventions_spark(raw):
    """Équivalent Spark de garage_cleaning.clean_interventions (mêmes règles)."""
    df = raw.dropDuplicates()

    amount_text = F.trim(F.regexp_replace(F.regexp_replace('montant_facture_eur', 'EUR', ''),
                                          ',', '.'))
    amount = amount_text.cast(DoubleType())
    # Piège Spark : 'nan EUR' devient le NOMBRE NaN (et non null) ; on le ramène à null
    # pour qu'il soit traité comme manquant, exactement comme dans pandas.
    df = df.withColumn('montant_facture_eur', F.when(F.isnan(amount), None).otherwise(amount))

    for column in ['origine', 'statut']:
        df = df.withColumn(column, F.regexp_replace(F.lower(F.trim(column)), r'\s+', ' '))

    df = df.withColumn('date_evenement', F.coalesce(
        *[F.to_timestamp('date_evenement', fmt) for fmt in DATE_FORMATS]))
    df = df.dropna(subset=['date_evenement'])

    df = df.withColumn('km_vehicule', F.when(F.col('km_vehicule') == KM_SENTINEL, None)
                       .otherwise(F.col('km_vehicule')).cast(DoubleType()))
    df = df.withColumn('duree_heures', F.when(F.col('duree_heures') < 0, None)
                       .otherwise(F.col('duree_heures')).cast(DoubleType()))
    df = df.withColumn('duree_atypique', F.col('duree_heures') > MAX_PLAUSIBLE_DURATION_H)
    df = df.withColumn('montant_manquant', F.col('montant_facture_eur').isNull())
    return df
