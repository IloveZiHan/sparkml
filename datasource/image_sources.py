
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("加载图像数据源") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

df = spark.read.format("image").option("dropInvalid", True).load("/Users/zhoufeng/Pictures/")

# 打印schema
df.printSchema()
df.show(truncate=True)
df.select("image.origin", "image.width", "image.height").show(truncate=False)
