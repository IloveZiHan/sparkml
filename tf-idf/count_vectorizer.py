from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TF-IDF") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

df = spark.createDataFrame([
    (0, "a b c d".split(" ")),
    (1, "a b b c d".split(" "))
], ["id", "words"])

cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

model = cv.fit(df)

result = model.transform(df)
result.show(truncate=False)