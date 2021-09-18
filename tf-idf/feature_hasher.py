from pyspark.ml.feature import FeatureHasher
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TF-IDF") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

dataset = spark.createDataFrame([
    (2.2, True, "1", "foo"),
    (3.3, False, "2", "bar"),
    (4.4, False, "3", "baz"),
    (5.5, False, "4", "foo")
], ["real", "bool", "stringNum", "string"])

hasher = FeatureHasher(inputCols=["real", "bool", "stringNum", "string"],
                       outputCol="features")

featurized = hasher.transform(dataset)
featurized.show(truncate=False)