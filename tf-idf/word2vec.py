from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Word2Vec") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=5, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)

result = model.transform(documentDF)
result.show(truncate=False)

for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))