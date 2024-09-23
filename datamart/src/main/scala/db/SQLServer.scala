package db

import org.apache.spark.sql.{DataFrame, SparkSession}

class SparkConfig(
    val appName: String,
    val deployMode: String,
    val driverMemory: String,
    val executorMemory: String,
    val executorCores: Int,
    val driverCores: Int
                 )

class DbConfig(
    val url: String,
    val user: String,
    val password: String,
    val driver: String
    )
    

class SQLServer(config: SparkConfig, val dbConfig: DbConfig) extends Database {
    private val spark = SparkSession.builder()
        .appName(config.appName)
        .master(config.deployMode)
        .config("spark.driver.cores", config.driverCores)
        .config("spark.executor.cores", config.executorCores)
        .config("spark.driver.memory", config.driverMemory)
        .config("spark.executor.memory", config.executorMemory)
        .config("spark.jars", "jars/mssql-jdbc-12.6.1.jre11.jar")
        .config("spark.driver.extraClassPath", "jars/mssql-jdbc-12.6.1.jre11.jar")
        .getOrCreate()

    override def getData(): DataFrame = {
        val jdbcOptions = Map(
        "url" -> dbConfig.url,
        "dbtable" -> "FoodProducts",
        "user" -> dbConfig.user,
        "password" -> dbConfig.password,
        "driver" -> dbConfig.driver
        )

        spark.read.format("jdbc").options(jdbcOptions).option("inferSchema", "true").load()
    }

    override def setPredictions(df: DataFrame): Unit = {
        val jdbcOptions = Map(
        "url" -> dbConfig.url,
        "dbtable" -> "FoodProducts",
        "user" -> dbConfig.user,
        "password" -> dbConfig.password,
        "driver" -> dbConfig.driver
        )

        df.write.mode("append").format("jdbc").options(jdbcOptions).save()
    }
}