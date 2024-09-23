from pyspark.sql import SparkSession, DataFrame
from src.configs import DatabaseConfig


class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        config.check_url()
        self.jdbc_options = {
            "url": self.config.url,
            "dbtable": "FoodProducts",
            "user": self.config.username,
            "password": self.config.password,
            "driver": self.config.driver,
        }

    def read_data(self, spark: SparkSession):
        return (
            spark.read.format("jdbc")
            .options(**self.jdbc_options)
            .option("inferSchema", "true")
            .load()
        )

    def write_data(self, df: DataFrame, mode="append"):
        df.write.format("jdbc").options(**self.jdbc_options).mode(mode).save()
