from src.clastering_process_spark import run
from src.configs import TrainConfig

train_config = TrainConfig()
run(train_config)
