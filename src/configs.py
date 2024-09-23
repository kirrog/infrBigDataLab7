from typing import Optional


class KMeansConfig:
    k: int = 2
    maxIter: int = 20
    seed: Optional[int] = None


class DataConfig:
    data_path: str = "data/openfood.csv"
    feature_path: str = "configs/features.json"


class SparkConfig:
    app_name: str = "food_cluster"
    deploy_mode: str = "local"
    driver_memory: str = "4g"
    executor_memory: str = "16g"
    executor_cores: int = 1
    driver_cores: int = 1


class DatabaseConfig:
    server: str = "mssql-server"
    database: str = "FoodData"
    username: str = "sa"
    password: str = "yourStrong(!)Password"
    driver: str = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    url: str = None

    def check_url(self):
        self.url = (f"jdbc:sqlserver://{self.server};databaseName={self.database};"
                    f"user={self.username};password={self.password};"
                    f"encrypt=false;trustServerCertificate=true")


class TrainConfig:
    kmeans: KMeansConfig = KMeansConfig()
    data: DataConfig = DataConfig()
    spark: SparkConfig = SparkConfig()
    db: DatabaseConfig = DatabaseConfig()
    save_to: str = "models/food_cluster"
