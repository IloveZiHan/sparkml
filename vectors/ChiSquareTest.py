from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

# 标签、特征
data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(0.5, 20.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (1.0, Vectors.dense(1.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]

spark = SparkSession.builder \
    .appName("卡方检验") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

df = spark.createDataFrame(data, ["label", "features"])

r = ChiSquareTest.test(df, "features", "label").head()

ChiSquareTest.test(df, "features", "label").show(truncate=False)

print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))