from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

if __name__ == "__main__":

    spark = SparkSession.builder.appName("SentimentTrain").getOrCreate()

    data = [(1, "Love this", 1.0), (2, "Hate this"), 0.0]
    df = spark.createDataFrame(data, ["id", "text", "label"])

    tokenizer = Tokenizer(inputCol="text", outputCol = "words")
    hashingTF = HashingTF(inputCol="words", outputCol = "rawFeatures")
    idf = IDF(inputCol = "rawFeatures", outputCol = "features")
    lr = LogisticRegression(maxIter = 10)

    pipeline = Pipeline(stages = [tokenizer, hashingTF, idf, lr])

    model = pipeline.fit(df)
    print("Spark Model Trained Successfully.")
