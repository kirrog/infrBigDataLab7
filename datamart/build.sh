SCALA_VERSION=2.12
DATAMART_VERSION=0.1.0-SNAPSHOT
JAR_SOURCE_PATH=target/scala-${SCALA_VERSION}/datamart_${SCALA_VERSION}-${DATAMART_VERSION}.jar

sbt clean && sbt package && cp ${JAR_SOURCE_PATH} ../jars/datamart.jar