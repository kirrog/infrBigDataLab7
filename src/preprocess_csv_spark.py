import json
import logging

import pyspark.sql
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions

FEATURES_COLUMN = "scaled_features"
logger = logging.Logger("preproc")


class Preprocessor:
    def __init__(self, spark_cs: pyspark.sql.SparkSession, features_path: str):
        with open(features_path, "r") as features_file:
            self.features = json.load(features_file)
        self.spark = spark_cs

    def preprocess(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        id_columns = self.features["id"]
        feature_numeric = self.features["numeric"]
        numeric_columns = [
            functions.col(c).cast("float").alias(c) for c in feature_numeric
        ]
        cat_columns = self.features["categorical"]

        all_columns = id_columns + numeric_columns + cat_columns
        df_with_selected_columns = df.select(*all_columns)
        # df_with_selected_columns.printSchema()

        vec_assembler = VectorAssembler(
            inputCols=feature_numeric, outputCol="features"
        )

        df_with_features = vec_assembler.transform(df_with_selected_columns)
        # df_with_features.printSchema()
        scaler = StandardScaler(inputCol="features", outputCol=FEATURES_COLUMN)
        scaler_model = scaler.fit(df_with_features)
        df_scaled_features = scaler_model.transform(df_with_features)
        # df_scaled_features.printSchema()
        return df_scaled_features
