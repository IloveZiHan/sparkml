from pyspark.ml.feature import Binarizer
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

continuousDataFrame = spark.createDataFrame([
    (0, 0.1),
    (1, 0.8),
    (2, 0.2)
], ["id", "feature"])

binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

binarizedDataFrame = binarizer.transform(continuousDataFrame)

print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
binarizedDataFrame.show()