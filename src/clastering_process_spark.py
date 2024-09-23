import logging

from pyspark.ml import clustering, evaluation
from pyspark.sql import SparkSession

from src import configs, preprocess_csv_spark, datamart

logger = logging.Logger("clustering")


def run(config: configs.TrainConfig):
    spark_config = config.spark
    spark = (
        SparkSession.builder.appName(spark_config.app_name)
        .master(spark_config.deploy_mode)
        .config("spark.driver.cores", spark_config.driver_cores)
        .config("spark.executor.cores", spark_config.executor_cores)
        .config("spark.driver.memory", spark_config.driver_memory)
        .config("spark.executor.memory", spark_config.executor_memory)
        .config("spark.jars", f"{config.datamart},jars/mssql-jdbc-12.6.1.jre11.jar")
        .config("spark.driver.extraClassPath", "jars/mssql-jdbc-12.6.1.jre11.jar")
        .getOrCreate()
    )

    # db_config = config.db
    # db_manager = database_manager.DatabaseManager(db_config)
    dm = datamart.DataMart(spark, config)
    df = dm.get_food()

    kmeans_kwargs = config.kmeans.__dict__
    logger.info("Using kmeans model with parameters: {}", kmeans_kwargs)
    logger.info("Training")
    model = clustering.KMeans(featuresCol=preprocess_csv_spark.FEATURES_COLUMN, **kmeans_kwargs)
    model_fit = model.fit(df)

    logger.info("Evaluation")
    evaluator = evaluation.ClusteringEvaluator(
        predictionCol="prediction",
        featuresCol=preprocess_csv_spark.FEATURES_COLUMN,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    output = model_fit.transform(df)
    output.show()

    score = evaluator.evaluate(output)
    logger.info("Silhouette Score: {}", score)

    logger.info("Saving to {}", config.save_to)
    model_fit.write().overwrite().save(config.save_to)

    logger.info("Writing result into DB")
    output = output.withColumn("prediction", output.prediction.cast("int"))
    dm.set_predictions(output.select("code", "prediction"))

    spark.stop()
    logger.info("Train successfully finished!")