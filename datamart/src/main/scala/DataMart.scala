import db.{Database, DbConfig, SQLServer, SparkConfig}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

class DataMart(sparkConfig: SparkConfig, dbConfig: DbConfig) {
    private val FEATURES_COLUMN = "scaled_features"
    private val COLUMNS = Map(
        "id" -> Array("code", "product_name"),
        "numeric" -> Array(
        "energy_kcal_100g",
        "energy_100g",
        "fat_100g",
        "saturated_fat_100g",
        "trans_fat_100g",
        "cholesterol_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "sodium_100g",
        "calcium_100g",
        "iron_100g",
        "nutrition_score_fr_100g"
        ),
        "categorical" -> Array("categories_en")
    )
    private val database: Database = new SQLServer(sparkConfig, dbConfig)

    private def preprocess(df: DataFrame): DataFrame = {
        val idColumns = COLUMNS.apply("id")
        val featureColumnNames = COLUMNS.apply("numeric")
        val featureColumns = featureColumnNames.map(c => col(c).cast("float"))
        val catColumns = COLUMNS.apply("categorical")

        val allColumns = idColumns.map(col) ++ featureColumns ++ catColumns.map(col)
        val dfWithSelectedColumns = df.select(allColumns: _*)

        val vac_assembler = new VectorAssembler().setInputCols(featureColumnNames).setOutputCol("features")
        val dfWithFeatures = vac_assembler.transform(dfWithSelectedColumns)

        val scaler = new StandardScaler().setInputCol("features").setOutputCol(FEATURES_COLUMN)
        val scalerModel = scaler.fit(dfWithFeatures)
        scalerModel.transform(dfWithFeatures)
    }

    def getFood(): DataFrame = {
        val data = database.getData()
        preprocess(data)
    }

    def setPredictions(df: DataFrame): Unit = {
        database.setPredictions(df.select("code", "prediction"))
    }
}