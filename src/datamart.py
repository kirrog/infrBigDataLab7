import pyspark
from src import configs


class DataMart:
    def __init__(self, spark: pyspark.sql.SparkSession, config: configs.TrainConfig):
        self.spark = spark
        sc = spark.sparkContext
        config.db.check_url()
        db_config = sc._jvm.db.DbConfig(
            config.db.url,
            config.db.username,
            config.db.password,
            config.db.driver
        )
        spark_config = sc._jvm.db.SparkConfig(
            config.spark.app_name,
            config.spark.deploy_mode,
            config.spark.driver_memory,
            config.spark.executor_memory,
            config.spark.executor_cores,
            config.spark.driver_cores,
        )
        self.datamart = sc._jvm.DataMart(spark_config, db_config)

    def get_food(self) -> pyspark.sql.DataFrame:
        jdf = self.datamart.getFood()
        return pyspark.sql.DataFrame(jdf, self.spark)

    def set_predictions(self, df: pyspark.sql.DataFrame):
        self.datamart.setPredictions(df._jdf)
