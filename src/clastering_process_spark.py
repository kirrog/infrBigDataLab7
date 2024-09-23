import logging

from pyspark.ml import clustering, evaluation
from pyspark.sql import SparkSession

from src import configs, preprocess_csv_spark, database_manager

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
        .config("spark.jars", "jars/mssql-jdbc-12.6.1.jre11.jar")
        .config("spark.driver.extraClassPath", "jars/mssql-jdbc-12.6.1.jre11.jar")
        .getOrCreate()
    )

    db_config = config.db
    db_manager = database_manager.DatabaseManager(db_config)

    preprocessor = preprocess_csv_spark.Preprocessor(spark, config.data.feature_path)
    data = db_manager.read_data(spark)
    # logger.warning(f"Rows count: {data.count()}")
    df = preprocessor.preprocess(data)

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
    db_manager.write_data(output.select("code", "prediction"), mode="append")
    logger.info("Train successfully finished!")
