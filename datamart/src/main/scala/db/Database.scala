package db

import org.apache.spark.sql.DataFrame
trait Database {
  def getData(): DataFrame
  def setPredictions(df: DataFrame): Unit
}