from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("线性回归测试") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

lr = LogisticRegression()

lr.fit()